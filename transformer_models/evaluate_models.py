import wandb
# from transformer_repetition_kit import count_parameters
# import logging

api_instance = wandb.Api()

# Get runs from the "witw/running_records" project but they need to have a test_f1 score
all_runs = api_instance.runs(path="witw/running_records")

for run in all_runs:
    if "test_f1" in run.summary and "trainable_parameters" not in run.summary:
        # Check to see if there is a model artifact
        run.file("output.log").download(replace=True)
        
        # Open the file and get the number of trainable parameters from the third
        # line
        with open("output.log", "r") as f:
            print("Updating run: ", run.name)
            trainable_parameters_line = ""
            for line in f.readlines():
                if line.startswith("The model has"):
                    trainable_parameters_line = line
                    break
            if trainable_parameters_line == "":
                exit(1)
            print(trainable_parameters_line)

            # Okay so the line will say something like The model has 11,208,455 trainable parameter
            # We want to get the number of parameters so we split on the space and take the last element
            # which is the number of parameters
            trainable_parameters = trainable_parameters_line.split(" ")[3]
            print(trainable_parameters)

            # So the number of parameters is a string so we need to convert it to an int
            # This number of parameters also has commas in it.
            # We need to remove the commas so we can convert it to an int
            trainable_parameters = trainable_parameters.replace(",", "")

            # Now, update the run record to say that the trainable parameters is the number
            # of parameters we just calculated
            run.summary["trainable_parameters"] = int(trainable_parameters)

            # Now, we need to save the run record so that the new summary is saved
            run.save()

            # Print the name of the run for tracking purposes
            print("Updated run: ", run.name)