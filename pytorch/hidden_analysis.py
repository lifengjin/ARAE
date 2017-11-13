from models import load_models
import argparse
import torch
import dill as pickle
from torch.autograd import Variable
from collections import Counter
import matplotlib
matplotlib.use('Agg')
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
from scipy import stats
from random import shuffle
from matplotlib.markers import MarkerStyle
from collections import namedtuple

parser = argparse.ArgumentParser(description='PyTorch ARAE for Text Eval')
parser.add_argument('--corpus_path', type=str, required=True,
                    help='directory to load corpus from')
parser.add_argument('--load_path', type=str, required=True,
                    help='directory to load model from')
parser.add_argument('--topk', type=int, default=10,
                    help='number of classes plotted')
parser.add_argument('--num-clusters', type=int, default=40, help='number of clusters of '
                                                                 'transformations')

args = parser.parse_args()
num_clusters = args.num_clusters
print(vars(args))

with open(args.corpus_path , 'rb') as b:
    corpus = pickle.load(b)

label_freqs = Counter(corpus.labels)
topk = label_freqs.most_common(args.topk)
topk = [x[0] for x in topk]

topk_data_set = []
topk_labels = []
topk_text = []

corpus_length = len(corpus.hiddens)

for i in range(corpus_length):
    if corpus.labels[i] in topk:
        # print(corpus.hiddens[i])
        topk_data_set.append(corpus.hiddens[i])
        topk_labels.append(corpus.labels[i])
        topk_text.append(' '.join(corpus.text[i]))
topk_data_set =torch.cat(topk_data_set, dim=0).numpy()

pca_transformer = PCA(2)
xys = pca_transformer.fit_transform(topk_data_set)
# print(xys.shape)

topk_labels = np.array(topk_labels)
topk_text = np.array(topk_text)
# print(topk_labels)

topk_labels_uniq = list(set(topk_labels))
# print(topk_labels_uniq)
# if True:
#     total_samples = len(topk_data_set)
#     num_chains = args.topk
#     colors_per_point = np.zeros(total_samples)
#     color_map = plt.get_cmap('rainbow')
#     cur_index = 0
#     # running_sum = num_sample_list[0]
#     all_markers = MarkerStyle.filled_markers
#     markers = ['' for i in range(total_samples)]
#     # chain_colors = []
#     chain_markers = []
#     chain_labels= []
#     print('number of chains: ', num_chains)
#
#     colors_per_point = ScalarMappable(cmap=color_map).to_rgba(range(args.topk))
#
#     patches = []
#     for i in range(len(chain_labels)):
#         patches.append(
#             Line2D(range(1), range(1), color='black', label=chain_labels[i], marker=chain_markers[i], markersize=2))
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     # num_sample_list.insert(0, 0)
#
#     for i in range(num_chains):
#         points = ax.scatter(xys[:, 0][topk_labels == top10_labels_uniq[i]],
#                             xys[:, 1][topk_labels == top10_labels_uniq[i]],
#                             c=colors_per_point[i], cmap=color_map
#                             , alpha=0.4, label='class_' + str(i))
#     ax.set_xlabel('1st Principle Component')
#     ax.set_ylabel('2nd Principle Component')
#     # ax.legend(patches, chain_labels)
#     ldg = ax.legend()
#     for handle in ldg.legendHandles:
#         handle._sizes = [30]
#     pp = PdfPages('vp_class_encoded' + '.pdf')
#     fig.savefig(pp, format='pdf')
#     pp.close()
#     # plt.cla()

# getting the transformations between hidden states within a label
# also get the estimates of variances of the gaussians
diffs = []
text_pairs = []
text_labels = []
topk_gaussian_means = []
topk_gaussian_vars = []
for label in topk_labels_uniq:
    labeled_hiddens = topk_data_set[topk_labels == label, ...]
    labeled_text = topk_text[topk_labels == label]
    # this_gaussian_mean = np.mean(labeled_hiddens, 0, keepdims=True)
    this_gaussian_var = np.var(labeled_hiddens, 0,keepdims=True)
    # print(this_gaussian_var.shape)
    topk_gaussian_vars.append(this_gaussian_var)
    # diag_vars = np.diag(this_gaussian_var.squeeze())
    # print(diag_vars.shape)
    # gaussian = stats.multivariate_normal(this_gaussian_mean.squeeze(), diag_vars)
    # for hidden in labeled_hiddens:
    #     print(hidden.shape)
    #     print(gaussian.cdf(hidden)**(1/300))
    for index1, row1 in enumerate(labeled_hiddens):
        for index2, row2 in enumerate(labeled_hiddens):
            if not np.array_equal(row1, row2):
                diffs.append(row2 - row1)
                text_pairs.append((labeled_text[index1], labeled_text[index2]))
                text_labels.append(label)

# calculate the average var of the gaussians
variances = np.mean(np.vstack(topk_gaussian_vars), 0)
cov = np.diag(variances)
print(variances)

# assert False

# clustering the transformations
kmeans = KMeans(n_clusters=num_clusters)
clusters = kmeans.fit_predict(diffs)
print(kmeans.cluster_centers_.shape)
counter = 40
counter_per_label = 4
label_counter = np.zeros(args.topk)
seen_texts = []
for i in range(num_clusters):
    this_counter = 0
    for index, cluster_assignment in enumerate(clusters):
        if cluster_assignment == i and label_counter[
            topk_labels_uniq.index(text_labels[index])] < counter_per_label and \
            text_pairs[index][0] not in seen_texts:
            # print(i, text_pairs[index][0], '\t', text_pairs[index][1])
            seen_texts.append(text_pairs[index][0])
            # seen_texts.append(text_pairs[index][1])
            this_counter += 1
            label_counter[
                topk_labels_uniq.index(text_labels[index])] += 1
        if this_counter == counter:
            label_counter = label_counter * 0
            shuffle(seen_texts)
            seen_texts = seen_texts[:len(seen_texts)//2]
            break

# saving the cluster centers
with open('cluster_centers.pkl', 'wb') as c:
    pickle.dump(kmeans.cluster_centers_, c)

cluster_centers = kmeans.cluster_centers_

all_data = torch.cat(corpus.hiddens, dim=0).numpy()
all_labels = np.array(corpus.labels)

generated_hiddens = []

LabeledHidden = namedtuple("LabeledHidden", ['label', 'hidden'])

# applying transformations to the hidden states of single instance labels
done_labels = 0

for label_index, label in enumerate(label_freqs):
    if label_freqs[label] == 1:
        true_hiddens = all_data[all_labels == label, ...]
        for hidden in true_hiddens:
            label_gaussian = stats.multivariate_normal(hidden.squeeze(), cov)
            for index, center in enumerate(cluster_centers):
                print(label, index)
                counter = 0
                while True:
                    new_sample = hidden+center
                    # if ave_prob > 0.4 and ave_prob < 0.9:
                    if counter == 0 or counter == 5 or counter == 20:
                        # ave_prob = label_gaussian.cdf(new_sample) ** (1 / 300)
                        # print(ave_prob)
                        generated_hiddens.append(LabeledHidden(label, hidden + center))

                    counter += 1
                    center = center * 0.9
                    if counter > 20:
                        break
        else:
            done_labels += 1
    if done_labels > 3:
        break

all_generated_hiddens = np.vstack([x.hidden for x in generated_hiddens])

all_generated_hiddens = Variable(torch.from_numpy(all_generated_hiddens).float(), volatile=True)

# load the decoder and generate from the transformed hiddens
model_args, idx2word, autoencoder, gan_gen, gan_disc \
    = load_models(args.load_path)

autoencoder.eval()

max_indices = autoencoder.generate(hidden=all_generated_hiddens,
                                   maxlen=20,
                                   sample=False)

max_indices = max_indices.data.cpu().numpy()
sentences = []
for idx in max_indices:
    # generated sentence
    words = [idx2word[x] for x in idx]
    # truncate sentences to first occurrence of <eos>
    truncated_sent = []
    for w in words:
        if w != '<eos>':
            truncated_sent.append(w)
        else:
            break
    sent = " ".join(truncated_sent)
    sentences.append(sent)

# saving out the generated sentences
with open('generated_sentences2.txt', 'w') as g:
    for index,s in enumerate(sentences):
        print(generated_hiddens[index].label, index%3, s, sep='\t', file=g)