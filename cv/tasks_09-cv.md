# Homework 9

1. Download the images for apartments by using scrapped urls from HM7.

2. Use pre-trained [MobileNet v2](https://pytorch.org/docs/stable/torchvision/models.html) to get embedding vector for apartment images. Feel free to use any other model :). Basically, this is feature extraction for next steps.

3. For image embeddings apply clustering algorithm in order to detect pattern/relations for images. Visualize and manually analyze the clusters. Add your observations into report (e.g. cluster one contains images related to kitchen).  

4. Add cluster number as a feature for the price prediction model from HM 8 and compare the results. Add information to report file.

5. Same as step 4, but consider using image embeddings as additional feature. Because there are multiple images per apartment, you can try out the following:
  - calculate the average embedding vector
  - concatenate the fixed number of image embeddings
  - randomly sample the fixed number of image embeddings

6. Don't forget to update your prediction endpoint with the new feature - image :)

# Deadline

**Due on 21.08.2019 23:59**
