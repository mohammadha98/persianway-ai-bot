import pandas as pd

# Read the Excel file
df = pd.read_excel('docs/product-qa-tickets.xls')

print('Sample questions:')
for i in range(5):
    print(f'{i+1}. {df.iloc[i]["Question"]}')

print('\nSample answer 1:')
print(df.iloc[0]['Answer'][:300])

print('\nSample answer 2:')
print(df.iloc[1]['Answer'][:300])

# Check if any questions are agriculture-related
agriculture_keywords = ['کود', 'خاک', 'کشاورزی', 'زراعت', 'گیاه', 'محصول']
agriculture_count = 0
for i, row in df.iterrows():
    question = str(row['Question']).lower()
    if any(keyword in question for keyword in agriculture_keywords):
        agriculture_count += 1
        if agriculture_count <= 3:  # Show first 3 agriculture-related questions
            print(f'\nAgriculture question {agriculture_count}: {row["Question"]}')

print(f'\nTotal agriculture-related questions: {agriculture_count} out of {len(df)}')