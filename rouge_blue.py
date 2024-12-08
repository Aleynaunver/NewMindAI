from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_rouge(reference, hypothesis):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def calculate_bleu(reference, hypothesis):
    
    reference_tokens = [reference.split()]
    hypothesis_tokens = hypothesis.split()
    
    # SmoothingFunction: for small data 
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothie)
    return bleu_score


"reference_text" #Since there were changes in the final case, no reference text was given. Therefore, I could only prepare the code.
hypothesis_text = "Spotify is a popular music streaming service that offers a vast library of songs, playlists, and radio stations. With a user-friendly interface and a wide range of features, it's no wonder that millions of users worldwide have fallen in love with it. Spotify's selection is vast, with millions of songs available to stream, including popular and obscure tracks. Users can create and manage their own playlists, discover new music through Discover Weekly and Release Radar, and even create custom playlists with specific moods or activities. Spotify's great app is available for both desktop and mobile devices, making it easy to access and enjoy music anywhere, anytime. However, Spotify has faced several issues in recent times, including being unable to play certain songs, having playlist shuffle and playback control issues, and sometimes disappearing songs from the playlist. Despite these problems, Spotify remains one of the best music apps available, offering a truly amazing music experience with its vast library, personalized recommendations, and social features."

#Calculate rouge score
rouge_scores = calculate_rouge("reference_text", hypothesis_text)
print("ROUGE Scores:")
print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")

#Calculate Blue score
bleu_score = calculate_bleu("reference_text", hypothesis_text)
print("\nBLEU Score:")
print(f"{bleu_score:.4f}")
