from data import prepare_data

def main():
    train_image_paths, train_image_captions, test_image_paths, test_image_captions, vocab = prepare_data()
    print("Train image paths:", len(train_image_paths))
    print("Train image captions:", len(train_image_captions))
    print("Test image paths:", len(test_image_paths))
    print("Test image captions:", len(test_image_captions))
    print("Vocab size:", len(vocab))
if __name__ == "__main__":
    main()