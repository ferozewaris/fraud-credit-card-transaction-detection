
from fraud_transaction_detection.pipeline.base import PipelineDriver


class DeploymentPipeline(PipelineDriver):

    def __init__(self, config, model, run_id):
        self.config = config
        self.model = model

    # TODO Implementation of model deployment
    def run(self):
        print("model is deployed")
