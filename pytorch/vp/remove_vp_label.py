fn = "wilkins_corrected.shuffled.51.sup_train.0"

new_fn = fn + ".nolabel"

with open(fn) as f, open(new_fn, 'w') as w:
    for line in f:
        print(line.split('\t')[0], file=w)