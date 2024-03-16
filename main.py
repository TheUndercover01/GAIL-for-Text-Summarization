from genrator import RolloutCreator
from discriminator import Discriminator
import tqdm
from data_util import PPORolloutStorage
from data_util import ArticleDataset
import torch
from config import Config
from losses import LossCalculator



def train_generator(model , rollout_creator  , store , opt , tbar , all_scores ):

    args = Config.geneartor_params()

    for i in range(args.epochs):#args.epochs

        # filling in the storage (phase 1)
        store.clear_history()
        rollouts, score = rollout_creator.make_experience(model, args.num_rollouts)
        store.push(rollouts)
        print("roll_out" , len(rollouts))
        train_dataloader = store.create_loader(args.batch_size, shuffle=True)

        #loss calculation and graident optimization (Phase 2)
        for batch in train_dataloader:
            for _ in range(args.ppo_epochs):
                loss, reward = LossCalculator.ppo_loss(batch)
                loss.backward()
                opt.step()
                opt.zero_grad()
                tbar.update()
        all_scores.append(score)
        tbar.set_description(f"| score: {score:.3f} |")


def RLGAF(model , reward_model , x , y , data , rollout_creator  , store , opt , tbar , all_scores):
    '''
    Here we need to make a training loop which consists of the following

    we need to firt train the reward model to do well using good and bad summaries.
    
    Then using that we need to train the generator to generate better summaries.

    We do this in a loop : x time the generator is trained and y times the discriminator.


    Problems to solve:

    Whihc model to pick for discriminator and genearator

    Applying LORA and QLORA.

    How to change the discriminator to outpout a score

    How to generatate bad and good summaries (then label and make a custom dataset to train discriminator to differenciate a good and bad summary (see custom_datset in discriminator))

    How long to train generator ( i.e what is x) and  How long to train discriminator ( i.e what is y)


    '''
    pass





if __name__ == '__main__' :
    '''
    calling all the functions necessary to run

    '''

    #data = #data
    #article_dataset = ArticleDataset(args.article_size , data)
    #store = PPORolloutStorage()
    #opt = torch.optim.AdamW(model.parameters(), args.lr)
    #total_steps = (args.num_rollouts//args.batch_size)*args.ppo_epochs*args.epochs
    #tbar = tqdm(initial=0, total=total_steps)
    # total_steps = (args.num_rollouts//args.batch_size)*args.ppo_epochs*args.epochs
    # tbar = tqdm(initial=0, total=total_steps)
    # all_scores = []
    # rollout_creator = RolloutCreator(article_dataset, article_batch_size=args.article_batch_size)
    #model = 
    # reward_model = 
    # 

    #RLGAF()


    pass