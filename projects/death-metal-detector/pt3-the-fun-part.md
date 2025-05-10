---
layout: page
title:
---
[<< Back to Detector](detector.md)<br>
[<< Back to Pt 1](pt1-getting-the-data.md)<br>
[<< Back to Pt 2](pt2-training-the-model.md)<br>

# Pt 3 - The Fun Part
In this section, we will...
- Load the trained model objects we saved in Pt 2.
- Write a function to
    - Transform lyrics into a format readable by the model
    - Estimate a score for the lyrics using the trained model
- Test the Function on 3 examples

## But first, a confession
I've been lying to you. The Death Metal Detector does nothing of the sort. It cannot reliably classify if lyrics came from a death metal song. Instead, given the subjective nature of music, and the spectrum of well and poorly fitting terms in the data, it uses the probability of death metal as a proxy for the song's "metalness" or "brutality".

Which, if you'll allow me to blast my own beat, I think is much more interesting. I can just google or ask ChatGPT if an artist is death metal. I trained a model to give me an *informed opinion*. It reflects my intuition and the cultural consensus, and yet it surprises me once in awhile. It reflects heavy metal culture earnestly, but in a way that can be kind of funny.

## Load Objects
For starters, let's load the Complement Naive Bayes classifier, the TF-IDF scorer, and the term dictionary/blacklist.


```python
import pickle
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

print("Loading objects...")
with open("CNB.pickle", "rb") as f:
    clf = pickle.load(f)

with open("TFIDF.pickle", "rb") as f:
    tfidf = pickle.load(f)

dictionary = pd.read_csv("dictionary.csv")
blacklist = pd.read_csv("blacklist.csv")

print("Complete")
```

    Loading objects...
    Complete
    

## Coding for user input
For the final application, I want a user to be able to supply their own lyrics, and get back the score that the trained classifier comes up with. Given a string of text, long or short, the following steps must take place:
1. The text is 'vectorized', meaning it is transformed into the Bag-of-Words format.
2. Any terms that match the blacklist (document frequency < 0.001 or > 0.6 in the training set) are removed.
3. Remaining terms are ["fuzzy-matched"](https://github.com/rapidfuzz/RapidFuzz?tab=readme-ov-file#scorers) matched against the dictionary
4. Term count data is transformed into a sparse array
5. Term counts are transformed into TF-IDF scores
6. The classifier returns a probability of 'death metal' based on the TF-IDF scores.


```python
import numpy as np

from rapidfuzz.process import cdist
from rapidfuzz.distance import Prefix
from scipy.sparse import csr_array

from sklearn.feature_extraction.text import CountVectorizer 

# Function to ingest lyrics and output a score
def death_metal_detector(lyrics):

    # Step 1: transform lyrics into bag of words
    vectorizer = CountVectorizer()
    Z = vectorizer.fit_transform([lyrics])
    
    Z_dataframe = pd.DataFrame(
        {"word_count": Z.toarray()[0],
         "word": vectorizer.get_feature_names_out()}
    )

    # Step 2: Remove blacklisted words
    feature_names = vectorizer.get_feature_names_out()
    feature_names = feature_names[ np.isin(feature_names, blacklist, invert=True) ]

    # Step 3: Match remaining words to their stems from the dictionary
    output = cdist(feature_names, dictionary["word"].values, 
                   scorer=Prefix.normalized_similarity,
                   score_cutoff=0.50001)
    
    test = pd.DataFrame(
        {
            "word": feature_names,
            "match": dictionary.iloc[ np.apply_along_axis(np.argmax, 1, output) ]["word"],
            "match_index": dictionary.iloc[ np.apply_along_axis(np.argmax, 1, output) ]["word_index"],
            "score": np.apply_along_axis(np.max, 1, output)
        }
    )
    
    test = test[test["score"] > 0.0]
    test = test.merge(Z_dataframe, on="word")
    
    Z_bow = dictionary.merge(test[["match", "word_count"]], how="left",
                     left_on="word", right_on="match").fillna(0.0)[["word_count", "word_index"]]
    
    # Step 4: Transform data to sparse array
    Z_bow["word_count"] = Z_bow["word_count"].astype("int")
    Z_bow["track_index"] = 0
    
    Z_coordinates = Z_bow[["word_count", "track_index", "word_index"]].values.T

    Z_csr = csr_array(
        (Z_coordinates[0], (Z_coordinates[1], Z_coordinates[2])),
        dtype=np.int32
    )

    # Step 5: Transform term counts to TF-IDF scores
    Z_tfidf = tfidf.transform(Z_csr)

    # Step 6: Estimate probability of Death Metal as a proxy for lyrical "brutality"
    
    # Note: Adjustment to final score based on word count and distinct word count
    w = len(lyrics.split())
    dw = len(feature_names)
        
    a = np.emath.logn(30, w)
    b = dw/w
    score_adjustment = np.mean([a,b])

    if score_adjustment > 1.0:
        score_adjustment = 1.0

    y_prob_tfidf = clf.predict_proba(Z_tfidf)[0][1]*score_adjustment
    
    # Complete!
    return(y_prob_tfidf)
    
```

## Testing the Function

Now that the function is complete, let's give it 3 different songs. 
- *Morbid Angel - Immortal Rites*, a definitive death metal song
- *Iron Maiden - Fear of the Dark*, a more "mainstream" heavy metal example
- *Taylor Swift - Cruel Summer*, a pop song with no ties to metal


```python
# Morbid Angel - Immortal Rites
immortal_rites = """
Gathered for a sacred rite
Subconscious minds allied
Call upon immortals
Call upon the oldest ones to intercede
Rid us of our human waste
Cleanse our earthly lives
Make us one with darkness
Enlighten us to your ways
From churning worlds of mindlessness
Come screams unheard before
Haunting voices fill the room
Their source remaining undefined
Shadows cast from faceless beings
Lost for centuries
Lords of death, I summon you
Reside within our brains
Cast your spells upon our lives
That we may receive
The gift of immortality
Bestowed on those who seek you
Gathered for a sacred rite
Subconscious minds allied
Call upon immortals
Call upon the oldest ones to intercede
Rid us of our human waste
Cleanse our earthly lives
Make us one with darkness
Enlighten us to your ways
Lords of death, I summon you
Reside within our brains
Cast your spells upon our lives
That we may receive
The gift of immortality
Bestowed on those who seek you
Now immortal
"""
```


```python
# Iron Maiden - Fear Of The Dark
fear_of_the_dark = """
I am a man who walks alone
And when I'm walking a dark road
At night or strolling through the park
When the light begins to change
I sometimes feel a little strange
A little anxious when it's dark
Fear of the dark, fear of the dark
I have a constant fear that something's always near
Fear of the dark, fear of the dark
I have a phobia that someone's always there
Have you run your fingers down the wall
And have you felt your neck skin crawl
When you're searching for the light?
Sometimes when you're scared to take a look
At the corner of the room
You've sensed that something's watching you
Fear of the dark, fear of the dark
I have a constant fear that something's always near
Fear of the dark, fear of the dark
I have a phobia that someone's always there
Have you ever been alone at night
Thought you heard footsteps behind
And turned around and no one's there?
And as you quicken up your pace
You find it hard to look again
Because you're sure there's someone there
Fear of the dark, fear of the dark
I have a constant fear that something's always near
Fear of the dark, fear of the dark
I have a phobia that someone's always there
Fear of the dark, fear of the dark
Fear of the dark, fear of the dark
Fear of the dark, fear of the dark
Fear of the dark, fear of the dark
Watching horror films the night before
Debating witches and folklores
The unknown troubles on your mind
Maybe your mind is playing tricks
You sense, and suddenly eyes fix
On dancing shadows from behind
Fear of the dark, fear of the dark
I have a constant fear that something's always near
Fear of the dark, fear of the dark
I have a phobia that someone's always there
Fear of the dark, fear of the dark
I have a constant fear that somethings always near
Fear of the dark, fear of the dark
I have a phobia that someone's always there
When I'm walking a dark road
I am a man who walks alone
"""
```


```python
# Taylor Swift - Cruel Summer
cruel_summer = """
Fever dream high in the quiet of the night
You know that I caught it (oh yeah, you're right, I want it)
Bad, bad boy, shiny toy with a price
You know that I bought it (oh yeah, you're right, I want it)
Killing me slow, out the window
I'm always waiting for you to be waiting below
Devils roll the dice, angels roll their eyes
What doesn't kill me makes me want you more
And it's new, the shape of your body
It's blue, the feeling I've got
And it's ooh, whoa-oh
It's a cruel summer
"It's cool, " that's what I tell 'em
No rules in breakable heaven
But ooh, whoa-oh
It's a cruel summer with you (yeah, yeah)
Hang your head low in the glow of the vending machine
I'm not dying (oh yeah, you're right, I want it)
You say that we'll just screw it up in these trying times
We're not trying (oh yeah, you're right, I want it)
So cut the headlights, summer's a knife
I'm always waiting for you just to cut to the bone
Devils roll the dice, angels roll their eyes
And if I bleed, you'll be the last to know, oh
It's new, the shape of your body
It's blue, the feeling I've got
And it's ooh, whoa-oh
It's a cruel summer
"It's cool, " that's what I tell 'em
No rules in breakable heaven
But ooh, whoa-oh
It's a cruel summer with you
I'm drunk in the back of the car
And I cried like a baby coming home from the bar (oh)
Said, "I'm fine, " but it wasn't true
I don't wanna keep secrets just to keep you
And I snuck in through the garden gate
Every night that summer, just to seal my fate (oh)
And I screamed, "For whatever it's worth
I love you, ain't that the worst thing you ever heard?"
He looks up, grinnin' like a devil
It's new, the shape of your body
It's blue, the feeling I've got
And it's ooh, whoa-oh
It's a cruel summer
"It's cool, " that's what I tell 'em
No rules in breakable heaven
But ooh, whoa-oh
It's a cruel summer with you
I'm drunk in the back of the car
And I cried like a baby coming home from the bar (oh)
Said, "I'm fine, " but it wasn't true
I don't wanna keep secrets just to keep you
And I snuck in through the garden gate
Every night that summer, just to seal my fate (oh)
And I screamed, "For whatever it's worth
I love you, ain't that the worst thing you ever heard?"
(Yeah, yeah, yeah, yeah)
"""
```


```python
lyrics_dict = {
    "Morbid Angel - Immortal Rites": immortal_rites,
    "Iron Maiden - Fear of the Dark": fear_of_the_dark,
    "Taylor Swift - Cruel Summer": cruel_summer
}

for name, lyrics in lyrics_dict.items():

    score = death_metal_detector(lyrics)
    
    print(name)
    print(f"{100*score:.1f}% BRUTAL")
    print("")
```

    Morbid Angel - Immortal Rites
    95.8% BRUTAL
    
    Iron Maiden - Fear of the Dark
    71.2% BRUTAL
    
    Taylor Swift - Cruel Summer
    1.7% BRUTAL
    
    

## A Reflection of Heavy Metal Culture

While genre is subjective, the amount of sources that categorize an artist as death metal increase the closer they fit a general consensus. That leads to the most intense and "brutal" lyrics being consistently associated with death metal, and as an artist moves down the spectrum to more accessible lyrics, the probability of that label goes down to 0.

The end result is a scale of "brutality" as understood by metal fans. Death metal staple *Morbid Angel* scores very high in this sense. *Iron Maiden*'s more accessible metal gets a C grade with their moderately dark themes. And, sorry Swifties, Taylor's songs might be brutal towards your emotions, but for this purpose she's barely on the radar.

If this was interesting, I encourage you to try it yourself, maybe with a music genre that you have a lot of expertise in.

[<< Back to Detector](detector.md)<br>
[<< Back to Pt 1](pt1-getting-the-data.md)<br>
[<< Back to Pt 2](pt2-training-the-model.md)<br>