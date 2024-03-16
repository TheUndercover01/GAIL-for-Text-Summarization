from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

class Discriminator():
    """
    Class for training the Discriminator.

    Args:
        model: The model to be trained.
        train_dataloader: Dataloader for training data.
        args: Arguments for configuring the trainer.

    Attributes:
        model: The model to be trained.
        train_dataloader: Dataloader for training data.
        args: Arguments for configuring the trainer.
    """

    def __init__(self, model, train_dataloader, args):
        """
        Initializes the Discriminator.

        Args:
            model: The model to be trained.
            train_dataloader: Dataloader for training data.
            args: Arguments for configuring the trainer.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.args = args

    def custom_dataset():
        '''

        make a datasetr for good summaries and bad summaries.
        
        labeling them and making them work to trin dfiscriminator

        '''
        pass

    def train(self):
        """
        Performs training of the model.

        Returns:
            List[float]: List of losses during training.
        """
        epoch = 1
        print_interval = self.args.logging_steps
        num_batches = len(self.train_dataloader)
        progress_bar = tqdm(total=num_batches * self.args.num_train_epochs, leave=True)
        progress_bar.set_description(f"| Train: Epoch {epoch}, evaluating ... |")
        losses = []
        temp_losses = []
        i = 0

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

        for epoch in range(1, self.args.num_train_epochs + 1):

            for batch in self.train_dataloader:
                chosen_input_ids = batch['chosen_input_ids'].to(self.model.device)
                chosen_attention_mask = batch['chosen_attention_mask'].to(self.model.device)
                rejected_input_ids = batch['rejected_input_ids'].to(self.model.device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(self.model.device)

                r_w = self.model(chosen_input_ids, attention_mask=chosen_attention_mask).logits #good summary
                r_l = self.model(rejected_input_ids, attention_mask=rejected_attention_mask).logits #bad summary

                loss = -F.logsigmoid(r_w - r_l).mean()

                # Accumulate the gradients
                loss /= self.args.gradient_accumulation_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if (i + 1) % self.args.gradient_accumulation_steps == 0 or i + 1 == len(self.train_dataloader):
                    optimizer.step()
                    optimizer.zero_grad()

                temp_losses.append(loss.item())
                if i % print_interval == 0:
                    progress_bar.set_description(f"| Train: Epoch {epoch}, loss = {loss.item():4f} |")
                    progress_bar.refresh()
                    losses.append(np.mean(temp_losses))
                    temp_losses = []
                progress_bar.update()
                i += 1

        progress_bar.set_description(f"| Train: Epoch {epoch}, loss = {loss.item():4f} |")
        progress_bar.refresh()

        return losses
