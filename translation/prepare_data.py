import random
import re


def split_data(path, new_name='casic2015', splits=4):
    with open(path) as f:
        content = f.readlines()
    random.seed(0)  # randomize same way every run
    random.shuffle(content)

    per_divide = len(content) // splits

    for i in range(splits):
        with open(new_name + '-' + str(i) + '.txt', mode='wt', encoding='utf-8') as file:
            file.write(''.join(content[per_divide * i: per_divide * (i + 1)]))


def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    return s


def load_pairs(filename):
    lines = open(filename).read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [normalize_string(l) for l in lines]
    return lines


def combine_data(path_1, path_2, path_new):
    old_pairs = load_pairs(path_1)
    new_pairs = load_pairs(path_2)

    combined_pairs = []

    for old_pair in old_pairs:
        if old_pair not in new_pairs:
            combined_pairs.append(old_pair)
    for new_pair in new_pairs:
        if new_pair not in old_pairs:
            combined_pairs.append(new_pair)

    with open(path_new, mode='wt', encoding='utf-8') as file:
        file.write('\n'.join(combined_pairs))


def combine_casict2015_files(path1, path2, path_new):
    lines1 = open(path1).read().strip().split('\n')
    lines2 = open(path2).read().strip().split('\n')
    lines_combied = []
    with open(path_new, mode='wt', encoding='utf-8') as file:
        for (line1, line2) in zip(lines1, lines2):
            # line2 = ' '.join(line2)  # TODO: only for chinese
            if len(line1.split(' ')) <= 25 and len(line2.split(' ')) <= 25:
                lines_combied.append(line1 + '\t' + line2 + '\n')
                file.write(line1 + '\t' + line2 + '\n')
    return lines_combied


# combined = combine_casict2015_files('../news-commentary-v12.de-en.en', '../news-commentary-v12.de-en.de', '../news-commentary-v12.de-en.de_total.txt')
#
# print(combined[0])
#


combined = []
with open("../rapid2016.de-en.en") as xh:
  with open('../rapid2016.de-en.de') as yh:
    with open("../rapid2016.de-en.de_total.txt",mode='wt', encoding='utf-8') as zh:
      #Read first file
      xlines = xh.readlines()
      #Read second file
      ylines = yh.readlines()
      assert(len(xlines) == len(ylines))
      #Combine content of both lists  and Write to third file
      for line1, line2 in zip(ylines, xlines):
        zh.write("{}\t{}\n".format(line1.strip(), line2.strip()))
        combined.append("{}\t{}\n".format(line1.strip(), line2.strip()))

f = open('../rapid2016.de-en.de_train.txt', 'w')
for line in combined[:int(len(combined) * 0.8)]:
    print(line)
    f.write(line)
f.close()

f = open('../rapid2016.de-en.de_test.txt', 'w')
for line in combined[int(len(combined) * 0.8):]:
    f.write(line)
f.close()
