import os
import time
import shutil
import luigi
from luigi.contrib.gcs import GCSClient, GCSTarget
from google.oauth2 import service_account
from sklearn.model_selection import train_test_split
from simpletransformers.language_modeling import LanguageModelingModel
import wandb
import logging

class DownloadDataset(luigi.Task):
    dataset_path = luigi.Parameter(default='motions.txt')

    def run(self):
        client = GCSClient()

        bucket = os.environ['GCS_BUCKET']
        fp = client.download(f"{bucket}/{self.dataset_path}")
        self.output().makedirs()

        os.replace(fp.name, self.output().path)

    def output(self):
        return luigi.LocalTarget('./data/motions.txt')

class SplitDataset(luigi.Task):
    test_size = luigi.FloatParameter(default=0.1)
    random_state = luigi.FloatParameter(default=12)

    def requires(self):
        return DownloadDataset()

    def run(self):
        with self.input().open('r') as f:
            motions = f.readlines()

        train, test = train_test_split(motions,
                                       test_size=self.test_size,
                                       random_state=self.random_state)

        with self.output()['train'].open('w') as f:
            f.writelines(train)

        with self.output()['test'].open('w') as f:
            f.writelines(test)



    def output(self):
        return {'train': luigi.LocalTarget('./data/train.txt'),
                'test': luigi.LocalTarget('./data/test.txt')}


class Train(luigi.Task):
    use_cuda = luigi.Parameter(default=False)
    train_batch_size = luigi.IntParameter(default=1)
    num_train_epochs = luigi.IntParameter(default=1)
    learning_rate = luigi.FloatParameter(default=5e-4)
    max_seq_length = luigi.IntParameter(default=256)

    def requires(self):
        return SplitDataset()

    def run(self):

        train_data = self.input()['train'].path
        test_data = self.input()['test'].path

        logging.basicConfig(level=logging.INFO)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)

        train_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "block_size": 256,
            "max_seq_length": 256,
            "learning_rate": self.learning_rate,
            "train_batch_size": self.train_batch_size,
            "evaluate_during_training":True,
            "save_model_every_epoch": False,
            "save_eval_checkpoints": False,
            "gradient_accumulation_steps": 1,
            "num_train_epochs": self.num_train_epochs,
            "mlm": False,
            "output_dir": f"./outputs/fine-tuned/",
            'fp16': False,
            'wandb_project': os.environ['WANDB_PROJECT']
        }

        wandb.init(config=train_args)

        model = LanguageModelingModel("gpt2", "gpt2", args=train_args, use_cuda=self.use_cuda)

        model.train_model(train_data, eval_file=test_data)

        wandb.log(train_args)
        wandb.join()

        model_zip = f"{time.strftime('%Y%m%d-%H%M%S')}"

        shutil.make_archive(model_zip, 'zip', './outputs/best_model')

        client = GCSClient()
        client.put(f"{model_zip}.zip", f"{os.environ['GCS_BUCKET']}/{model_zip}.zip")


if __name__ == '__main__':
    luigi.run()
