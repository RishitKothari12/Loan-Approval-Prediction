import pickle
import gzip

# Load and inspect the file contents
with gzip.open("data.pkl.gz", "rb") as f:
    model_data = pickle.load(f)

print(type(model_data))  # Check the type of the loaded object
print(model_data)  # Print the contents to understand its structure

if isinstance(model_data, dict):
    print(model_data.keys())  # Print keys to identify model location

# Assuming `model` is your trained model object
with gzip.open("data.pkl.gz", "wb") as f:
    pickle.dump(Loan_Approval_Data_Train, f)
