import csv
import xml.etree.ElementTree as ET
import yaml
import os
import argparse


def traverse_tree(reviews):
    review_cnt = 0
    all_review = []
    for review in reviews:
        if (review.tag == 'Review'):
            for sentences in review:
                if (sentences.tag == 'sentences'):
                    for sentence in sentences:
                        cmt = []
                        if (sentence.tag == 'sentence'):
                            review_cnt += 1
                            for child in sentence:
                                if (child.tag == 'text'):
                                    cmt.append(child.text)

                                if (child.tag == 'Opinions'):
                                    target = ''
                                    category = ''
                                    polarity = ''
                                    from_ = ''
                                    to_ = ''
                                    for opinion in child:
                                        target += opinion.attrib['target'] + '~'
                                        category += opinion.attrib['category'] + '~'
                                        polarity += opinion.attrib['polarity'] + '~'
                                        from_ += opinion.attrib['from'] + '~'
                                        to_ += opinion.attrib['to'] + '~'
                                    cmt.append(target)
                                    cmt.append(category)
                                    cmt.append(polarity)
                                    cmt.append(from_)
                                    cmt.append(to_)

                        all_review.append(cmt)

    print("Number review:", review_cnt, len(all_review))
    return all_review


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Chuyển đổi XML sang CSV')

    parser.add_argument('--domain', '--dm', type=str, required=True,
                        help='Tên domain', choices=['restaurant', 'hotel'])

    parser.add_argument('--languague', '--lang', type=str,
                        required=True, help='Ngôn ngữ', choices=['en', 'dutch', 'fr', 'rus', 'spa', 'tur'])

    parser.add_argument('--type', '--tp', type=str,
                        required=True, help='Train/Dev/Test', choices=['train', 'dev', 'test'])

    args = parser.parse_args()

    domain = args.domain
    lang = args.languague
    typeXML = args.type + '_xml'
    typeCSV = args.type + '_csv'

    with open('semeval/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    path_xml = config[domain][lang][typeXML]
    path_csv = config[domain][lang][typeCSV]

    tree = ET.parse(path_xml)
    root = tree.getroot()

    all_review = traverse_tree(root)

    if not os.path.exists(os.path.dirname(path_csv)):
        os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    with open(path_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            ['text', 'target', 'category', 'polarity', 'from', 'to'])
        for review in all_review:
            csvwriter.writerow(review)
