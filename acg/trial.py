from config import config

print(config.model.type)          # Output: "resnet50"
print(config.model.learning_rate) # Output: 0.001
print(config.data.train_path)     # Output: "/data/train"
print(config.experiment.name)     # Output: "baseline"

print(config.model.llm.name)
