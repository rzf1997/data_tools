import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm


human_keywords = ['Person', 'Individual', 'Human', 'Child', 'Adult', 'Teenager', 'Senior', 'Infant', 'Parent', 'Sibling', 'Friend', 'Neighbor', 'Citizen', 'Foreigner', 'Immigrant', 'Worker', 'Employee', 'Employer', 'Boss', 'Manager', 'Colleague', 'Partner', 'Spouse', 'Husband', 'Wife', 'Boyfriend', 'Girlfriend', 'Fiancé', 'Fiancée', 'Relative', 'Ancestor', 'Descendant', 'Twin', 'Triplet', 'Roommate', 'Classmate', 'Teammate', 'Leader', 'Follower', 'Mentor', 'Student', 'Teacher', 'Professor', 'Doctor', 'Nurse', 'Patient', 'Customer', 'Client', 'Artist', 'Musician', 'Actor', 'Writer', 'Poet', 'Athlete', 'Player', 'Fan', 'Spectator', 'Judge', 'Lawyer', 'Police', 'Firefighter', 'Soldier', 'Pilot', 'Engineer', 'Scientist', 'Researcher', 'Inventor', 'Designer', 'Architect', 'Chef', 'Cook', 'Baker', 'Driver', 'Mechanic', 'Farmer', 'Fisherman', 'Merchant', 'Salesperson', 'Cashier', 'Librarian', 'Author', 'Journalist', 'Photographer', 'Filmmaker', 'Director', 'Producer', 'Editor', 'Publisher', 'Critic', 'Psychologist', 'Therapist', 'Counselor', 'Advisor', 'Consultant', 'Volunteer', 'Activist', 'Politician', 'Voter', 'Leader', 'Follower', 'Girl', 'Boy', 'Man', 'Men', 'Woman', 'Women', 'People', 'Folks', 'Guys', 'Ladies', 'Gent', 'Children', 'Citizens']

# human_keywords = ['Eat', 'Eating', 'Eats', 'Eaten', 'Ate']
# human_keywords = ["Talk", "Talked", "Talked", "Talking", "Talks", "Listen", "Listens", "Listening", "Listened", "Listened", "Speak", "Speaks", "Speaking", "Spoke", "Spoken"]
# human_keywords = ["Walk", "Walks", "Walking", "Walked", "Walked"]

lower_human_keywords = [keyword.lower() for keyword in human_keywords]

key_words = human_keywords + lower_human_keywords

dataset_list = [
    '/project/llmsvgen/share/opensora_datafile/artgrid_human_single_info.csv',
    '/project/llmsvgen/share/opensora_datafile/movies_human_single_info.csv',
]

data_list = []
c = 0
for dataset in dataset_list:
    csv_file = pd.read_csv(dataset)
    for idx, row in tqdm(csv_file.iterrows(), total=csv_file.shape[0]):
        # if c > 400000:
        #     break
        caption_list = row['text'].split(" ")
        if any(word in caption_list for word in key_words):
            path = row['path']
            text = row['text']
            fps = row['fps']
            frames = row['frames']
            height = row['height']
            width = row['width']
            text_long = row['text_long'] if 'text_long' in row else None
            data_list.append([path, text, fps, frames, height, width, text_long])
            c += 1

df = pd.DataFrame(data_list, columns=['path', 'text', 'fps', 'frames', 'height', 'width', 'text_long'])
df.to_csv('/project/llmsvgen/share/opensora_datafile/human_single_walk_info.csv', index=False)

print(f"Total number of videos: {c}")
# print(f"Number of videos with human keywords: {df.shape[0]}")
