import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import LabelEncoder

copy_data = pd.read_csv("datasets/copy_data.csv")

class sentence_similarity:
    
    #Grouping function by threshold
    def test_thresholds(reviews, embeddings, thresholds):

        results = {}    #empty dictionary for each threshold

        for threshold in thresholds:

            #cosine similarity matrix for embeddings
            cosine_similarities = util.pytorch_cos_sim(embeddings, embeddings)
            visited = set()
            group_count = 0

            for i, row in enumerate(cosine_similarities):

                if i not in visited:
                    
                    #Find indices of reviews similar to the current one based on the threshold
                    similar_reviews = [j for j, sim in enumerate(row) if sim > threshold and i != j]
                
                    if similar_reviews:
                        visited.update(similar_reviews)
                        visited.add(i)
                        group_count += 1

            results[threshold] = group_count

        return results


    def handle_similar_sentences(data):

        # Sentence Similarity
        similarity_model = SentenceTransformer("all-MiniLM-L6-v2")  #load Sentence Transformer model


        label_encoder = LabelEncoder()
        copy_data["encoded_label"] = label_encoder.fit_transform(copy_data["label"])



        #Split positive and negative comments
        positive_reviews = data[data["encoded_label"] == 1]["cleaned_review"].tolist()
        negative_reviews = data[data["encoded_label"] == 0]["cleaned_review"].tolist()

        #Creating embeddings
        positive_embeddings = similarity_model.encode(positive_reviews, convert_to_tensor=True)
        negative_embeddings = similarity_model.encode(negative_reviews, convert_to_tensor=True)

        
        thresholds_to_test = [0.90, 0.94, 0.98, 0.999]

        threshold_results_pos= sentence_similarity.test_thresholds(positive_reviews, positive_embeddings, thresholds_to_test)
        threshold_results_neg = sentence_similarity.test_thresholds(negative_reviews, negative_embeddings, thresholds_to_test)

        #Number of groups for each threshold tested
        for threshold, group_count in threshold_results_pos.items():           #for positive groups
            print(f"Threshold: {threshold}, Groups number: {group_count}")


        for threshold, group_count in threshold_results_neg.items():
            print(f"Threshold: {threshold}, Groups number: {group_count}")     #for negative groups


        selected_threshold_for_pos = 0.999
        selected_threshold_for_neg = 0.94


        #Number of groups taken with the selected threshold 
        selected_groups_pos = threshold_results_pos[selected_threshold_for_pos]
        selected_groups_neg = threshold_results_neg[selected_threshold_for_neg]

        #Selecting positive and negative reviews with threshold
        positive_reviews_selected = positive_reviews[:selected_groups_pos]     #selected positive reviews with threshold
        negative_reviews_selected2 = negative_reviews[:selected_groups_neg]    #selected negative reviews with threshold

        #Selecting positive and negative embeddings with threshold
        positive_embeddings_selected = positive_embeddings[:selected_groups_pos]
        negative_embeddings_selected = negative_embeddings[:selected_groups_neg]


        combined_reviews = positive_reviews_selected + negative_reviews_selected2                             #the reviews
        combined_embeddings = positive_embeddings_selected.tolist() + negative_embeddings_selected.tolist()   #group labels assigned to comments

        #Determine groups for each review
        positive_groups = [f'Positive Group {i}' for i in range(selected_groups_pos)]
        negative_groups = [f'Negative Group {i}' for i in range(selected_groups_neg)]

        combined_groups = positive_groups + negative_groups        

        df_combined = pd.DataFrame({
            'Review': combined_reviews,
            'Group': combined_groups
        })

        print(df_combined.head())
        df_combined.to_csv('similarity_reviews_groups.csv', index=False)