import pandas as pd

# 1. During your test loop, collect all info into a list
results_list = []

for batch in test_loader:
    inputs, labels, metadata = batch # Assuming your loader provides metadata
    outputs = model(inputs)
    probs = F.softmax(outputs, dim=1)
    conf, pred = torch.max(probs, dim=1)
    
    # Apply your collaboration policy (Strategy B)
    decision = "HUMAN" if (1 - probs[:, 0].item()) > 0.15 else "AI"

    results_list.append({
        "subject_id": metadata['id'],
        "gender": metadata['gender'], 
        "age": metadata['age'],
        "y_true": labels.item(),
        "y_pred": pred.item(),
        "confidence": conf.item(),
        "decision": decision
    })

# 2. Convert to a DataFrame
df = pd.DataFrame(results_list)
df.to_csv("test_results_with_metadata.csv", index=False)

# 3. THE SUBGROUP ANALYSIS (The Fairness Check)
print("\n--- FAIRNESS ANALYSIS BY GENDER ---")
# Compare the Escalation Rate (how often humans are called)
escalation_by_gender = df.groupby('gender')['decision'].apply(lambda x: (x == 'HUMAN').mean() * 100)
print(f"Escalation Rate (%): \n{escalation_by_gender}")

# Compare Accuracy by gender
accuracy_by_gender = df.groupby('gender').apply(lambda x: (x['y_true'] == x['y_pred']).mean())
print(f"\nAI Accuracy: \n{accuracy_by_gender}")