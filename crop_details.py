import pandas as pd
import pickle

# Load crop details dataset
data = pd.read_csv("Dataset/eggplant_details.csv")  # Update path if needed

print("Dataset loaded successfully!")
print(data.head())

def get_crop_info(crop_name):
    """
    Returns all matching records for the given crop name from the dataset,
    handling whitespace and case sensitivity robustly.
    """
    crop_name_clean = crop_name.strip().lower()
    # For debugging: print all names
    print("Searching for:", crop_name_clean)
    print("Available names:", data["Crop Name"].astype(str).str.lower().str.strip().tolist())
    result = data[data["Crop Name"].astype(str).str.lower().str.strip() == crop_name_clean]
    if result.empty:
        return f"No records found for crop: {crop_name}"
    else:
        return result.to_dict(orient="records")

# Example usage:
if __name__ == "__main__":
    crop = "Pusa Uttam"
    info = get_crop_info(crop)
    print(info)

# Save the dataset as a pickle model for deployment (optional)
pickle.dump(data, open("crop_details_model.pkl", "wb"))
