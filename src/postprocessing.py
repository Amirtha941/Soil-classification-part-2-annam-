
"""
Author: Annam.ai IIT Ropar  
Team Name: AgroMinds AI
Team Members: amirtha k,Tharun babu C
Leaderboard Rank:75

"""

import pandas as pd
import numpy as np

# Example label map (modify as per your class encoding)
label_map = {
    0: "alluvial soil",
    1: "red soil",
    2: "black soil",
    3: "clay soil"
}

def postprocessing(predictions, image_ids, output_path="submission.csv"):
  
    print("Running postprocessing...")

    # Map predicted indices to class labels
    predicted_labels = [label_map[int(pred)] for pred in predictions]

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        "image_id": image_ids,
        "label": predicted_labels
    })

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved to: {output_path}")

    return 0
