from datasets import Dataset, load_dataset

def get_1k_examples(ds):
    examples = []
    for example in ds:
        examples.append(example)
        if len(examples) >= 1000:
            break
    return examples

def main():
    coco_ds = load_dataset("RIW/small-coco", split="validation", streaming=True)
    coco_ex = get_1k_examples(coco_ds)

    k = 20
    topk = [88, 539, 87, 36, 13, 538, 75, 484, 74, 500, 33, 89, 72, 99, 16, 540, 34, 73, 114, 101]
    bottomk = [178, 198, 177, 4, 829, 497, 7, 830, 828, 3, 479, 831, 508, 478, 498, 291, 509, 480, 453, 180]

    print('TOP')
    for i in range(k):
        ex = coco_ex[topk[i]]
        print(ex['caption'], ex['url'])

    print('BOTTOM')
    for i in range(k):
        ex = coco_ex[bottomk[i]]
        print(ex['caption'], ex['url'])

if __name__ == "__main__":
    main()