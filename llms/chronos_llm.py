import io
import os
import yaml
import torch
from ts_llm import TimeSeriesLLM
from chronos import ChronosPipeline

class ChronosLLM(TimeSeriesLLM):
    def __init__(
            self,
            settings={
                'model': "amazon/chronos-t5-base",
                'device_map': "cuda",
                'torch_dtype': torch.float32
            },
            log_dir="./logs",
            name = "chronos_llm"
    ):
        super(ChronosLLM, self).__init__(name=name)
        self.llm_model = ChronosPipeline.from_pretrained(
            model=settings['model'],
            device_map=settings['device_map'],
            torch_dtype=settings['torch_dtype']
        )
        self._llm_settings = settings
        self._log_dir = log_dir

    def load_ckpt(self, ckpt_path):
        self.llm_model = ChronosPipeline.from_pretrained(
            model=ckpt_path,
            device_map=self._llm_settings['device_map'],
            torch_dtype=self._llm_settings['torch_dtype']
        )

    def predict(
            self,
            test_data,
            settings={
                'prediction_length': 64,
                'num_samples': 100,
            }
    ):
        # TODO: batch test data -> all sample simultaneously is too large for memory
        # TODO: If sequence length is too long, split it into smaller sequences and predict auto-regressively
        llm_prediction = self.llm_model.predict(
            context=test_data,
            prediction_length=settings['prediction_length'],
            num_samples=settings['num_samples'],
            limit_prediction_length=False
        )

        return llm_prediction

    def train(
            self,
            train_data_path,
            settings={
                'max_steps': 10000,
                'random_init': True,
                'save_steps': 1000,
                'prediction_length': 200
            }
    ):
        with open("./chronos-forecasting/scripts/training/configs/{}.yaml".format(self._llm_settings['model'].split("/")[-1]), 'r') as stream:
            data_loaded = yaml.safe_load(stream)

        data_loaded['model_id'] = self._llm_settings['model']
        data_loaded['training_data_paths'] = [train_data_path]
        data_loaded['probability'] = [1.0]
        data_loaded['max_steps'] = settings['max_steps']
        data_loaded['random_init'] = settings['random_init']
        data_loaded['save_steps'] = settings['save_steps']
        data_loaded['prediction_length'] = settings['prediction_length']
        output_dir = self._log_dir + "/{}".format(self._llm_settings['model'].split("/")[-1])
        data_loaded['output_dir'] = output_dir
        # store yaml file
        with io.open('./chronos-forecasting/scripts/training/configs/{}_custom.yaml'.format(self._llm_settings['model'].split("/")[-1]), 'w',
                     encoding='utf8') as outfile:
            yaml.dump(data_loaded, outfile)

        os.system("python ./chronos-forecasting/scripts/training/train.py --config ./chronos-forecasting/scripts/training/configs/{}_custom.yaml".format(self._llm_settings['model'].split("/")[-1]))

        return output_dir

