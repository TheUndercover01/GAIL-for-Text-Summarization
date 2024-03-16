from omegaconf import DictConfig

class Config():
        
    def geneartor_params(self):
        """
        Returns configuration parameters for the generator model.

        Returns:
            DictConfig: Configuration parameters for the generator.
        """
        self.genargs = {
            'generator_model_name_or_path': 'EKKam/opt-1.5b_imdb_sft',
            'reward_model_name_or_path': 'EKKam/opt350m_imdb_sentiment_reward',
            'seq_length': 1024,
            'batch_size': 64,
            'lr': 0.00006,
            'prompt_size': 30,
            'prompt_batch_size': 128,
            'num_rollouts': 128,
            'epochs': 100,
            'ppo_epochs': 4,
            'gen_kwargs': {
                'max_new_tokens': 40,
                'top_k': 0,
                'top_p': 1.0,
                'do_sample': True
            },
            'kl_coef': 0.01,
            'gamma': 1,
            'lam': 0.95,
            'cliprange': 0.2,
            'cliprange_value': 0.2,
            'vf_coef': 1,
        }

        genargs = DictConfig(genargs)
        return genargs
    
    def discriminator_params(self):
        """
        Returns configuration parameters for the discriminator model.

        Returns:
            DictConfig: Configuration parameters for the discriminator.
        """
        self.disargs = {
            "seed": 42,
            'model_name_or_path': 'facebook/opt-350m',
            'learning_rate': 5e-5,
            'batch_size': 2,
            'gradient_accumulation_steps': 16,
            'num_train_epochs': 1,
            'num_workers': 10,
            'seq_length': 1024,
            'logging_steps': 10,
        }

        disargs = DictConfig(disargs)
        return disargs



    