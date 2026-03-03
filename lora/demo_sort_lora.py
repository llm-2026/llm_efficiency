import pickle

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed
from lora import GPT_LoRA, get_lora_model

set_seed(3407)


class SortDataset(Dataset):
    """
    Dataset for the Sort problem. E.g. for problem length 6:
    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
    Which will feed into the transformer concatenated as:
    input:  0 0 2 1 0 1 0 0 0 1 1
    output: I I I I I 0 0 0 1 1 2
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(self, split, length=6, num_digits=3):
        assert split in {"train", "test"}
        self.split = split
        self.length = length
        self.num_digits = num_digits

    def __len__(self):
        return 10000  # ...

    def get_vocab_size(self):
        return self.num_digits

    def get_block_size(self):
        # the length of the sequence that will feed into transformer,
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length * 2 - 1

    def __getitem__(self, idx):

        # use rejection sampling to generate an input example from the desired split
        while True:
            # generate some random integers
            inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
            # half of the time let's try to boost the number of examples that
            # have a large number of repeats, as this is what the model seems to struggle
            # with later in training, and they are kind of rate
            if torch.rand(1).item() < 0.5:
                if inp.unique().nelement() > self.length // 2:
                    # too many unqiue digits, re-sample
                    continue
            # figure out if this generated example is train or test based on its hash
            h = hash(pickle.dumps(inp.tolist()))
            inp_split = (
                "test" if h % 4 == 0 else "train"
            )  # designate 25% of examples as test
            if inp_split == self.split:
                break  # ok

        # solve the task: i.e. sort
        sol = torch.sort(inp)[0]

        # concatenate the problem specification and the solution
        cat = torch.cat((inp, sol), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input locations
        y[: self.length - 1] = -1
        return x, y


# print an example instance of the dataset
train_dataset = SortDataset("train")
test_dataset = SortDataset("test")

model_config = GPT.get_default_config()
model_config.model_type = "gpt-nano"
model_config.vocab_size = train_dataset.get_vocab_size()
model_config.block_size = 24
model_config.lora_rank = 8
model_config.lora_alpha = 32
model_config.lora_dropout = 0
model = GPT_LoRA(model_config)

# create a Trainer object
train_config = Trainer.get_default_config()
train_config.learning_rate = (
    5e-4  # the model we're using is so small that we can go a bit faster
)
train_config.max_iters = 1000
train_config.num_workers = 0
trainer = Trainer(train_config, model, train_dataset)


def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(
            f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"
        )


trainer.set_callback("on_batch_end", batch_end_callback)

trainer.run()

model.eval()

dataset = {"train": train_dataset, "test": test_dataset}


def eval_split(trainer, split, max_batches, dataset=dataset):
    dataset = dataset[split]
    n = dataset.length  # naugy direct access shrug
    results = []
    mistakes_printed_already = 0
    loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
    for b, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        y = y.to(trainer.device)
        # isolate the input pattern alone
        inp = x[:, :n]
        sol = y[:, -n:]
        # let the model sample the rest of the sequence
        cat = model.generate(
            inp, n, do_sample=False
        )  # using greedy argmax, not sampling
        sol_candidate = cat[:, -n:]  # isolate the filled in sequence
        # compare the predicted sequence to the true sequence
        correct = (
            (sol == sol_candidate).all(1).cpu()
        )  # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            if (
                not correct[i] and mistakes_printed_already < 3
            ):  # only print up to 3 mistakes to get a sense
                mistakes_printed_already += 1
                print(
                    "GPT claims that %s sorted is %s but gt is %s"
                    % (inp[i].tolist(), sol_candidate[i].tolist(), sol[i].tolist())
                )
        if max_batches is not None and b + 1 >= max_batches:
            break
    rt = torch.tensor(results, dtype=torch.float)
    print(
        "%s final score: %d/%d = %.2f%% correct"
        % (split, rt.sum(), len(results), 100 * rt.mean())
    )
    return rt.sum()


# run a lot of examples from both train and test through the model and verify the output correctness
with torch.no_grad():
    train_score = eval_split(trainer, "train", max_batches=50)
    test_score = eval_split(trainer, "test", max_batches=50)

# Now we modify the distribution of the dataset a little bit and use LoRA to fine-tune.
train_dataset2 = SortDataset("train", length=10)
test_dataset2 = SortDataset("test", length=10)

dataset2 = {"train": train_dataset2, "test": test_dataset2}
with torch.no_grad():
    train_score = eval_split(trainer, "train", max_batches=50, dataset=dataset2)
    test_score = eval_split(trainer, "test", max_batches=50, dataset=dataset2)

###
# your code herefor training with LoRA
###

model.eval()
with torch.no_grad():
    train_score = eval_split(trainer, "train", max_batches=50, dataset=dataset2)
    test_score = eval_split(trainer, "test", max_batches=50, dataset=dataset2)
