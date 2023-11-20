import runpod
import generate_appeal

generator = generate_appeal.AppealGenerator()


runpod.serverless.start({"handler": generator.generate_appeal})
