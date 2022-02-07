# BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese

BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese
Nguyen Luong Tran, Duong Minh Le, Dat Quoc Nguyen
VinAI Research, Hanoi, Vietnam
{v.nguyentl12, v.duonglm1, v.datnq9}@vinai.io
Abstract
In this paper, we present BARTpho with two
versions BARTphosyllable and BARTphoword ,
which are the first public large-scale monolin-
gual sequence-to-sequence models pre-trained
for Vietnamese. BARTpho uses the “large”
architecture and the pre-training scheme of
the sequence-to-sequence denoising autoen-
coder BART, thus especially suitable for gen-
erative NLP tasks. We conduct experiments
to compare our BARTpho with its competi-
tor mBART on a downstream task of Viet-
namese text summarization and show that: in
both automatic and human evaluations, BART-
pho outperforms the strong baseline mBART
and improves the state-of-the-art. We re-
lease BARTpho to facilitate future research
and applications of generative Vietnamese
NLP tasks. Our BARTpho models are publicly available at: https://github.com/VinAIResearch/BARTpho.

## 1 Introduction
The masked language model BERT (Devlin et al.,
2019) and its variants, pre-trained on large-scale
corpora, help improve the state-of-the-art (SOTA)
performances of various natural language under-
standing (NLU) tasks. However, due to a bidi-
rectionality nature, it might be difficult to di-
rectly apply those pre-trained language models
to natural language generation tasks (Wang and
Cho, 2019). Therefore, pre-trained sequence-to-
sequence (seq2seq) models are proposed to handle
this issue (Dong et al., 2019; Lewis et al., 2020;
Zhang et al., 2020; Raffel et al., 2020; Qi et al.,
2020; Xue et al., 2021a). The success of these pre-
trained seq2seq models has largely been limited
to the English language. From a societal, cultural,
linguistic, cognitive and machine learning perspec-
tive (Ruder, 2020), it is worth investigating pre-
trained seq2seq models for languages other than
English. For other languages, one could employ
existing pre-trained multilingual seq2seq models
(Liu et al., 2020; Xue et al., 2021b; Qi et al., 2021)
or retrain language-specific models using the pro-
posed seq2seq architectures (Eddine et al., 2020;
Shao et al., 2021). Note that retraining a language-
specific model might be preferable as dedicated
language-specific models still outperform multilin-
gual ones (Nguyen and Nguyen, 2020).
Regarding Vietnamese, to the best of our knowl-
edge, there is not an existing monolingual seq2seq
model pre-trained for Vietnamese. In addition,
another concern is that all publicly available pre-
trained multilingual seq2seq models are not aware
of the linguistic characteristic difference between
Vietnamese syllables and word tokens. This
comes from the fact that when written in Viet-
namese, in addition to marking word boundaries,
the white space is also used to separate sylla-
bles that constitute words.1 For example, 7-
syllable written text “chúng tôi là những nghiên cứu
viên”we are researchers forms 4-word text “chúng_tôiwe
làare những nghiên_cứu_viênreseacher ”. Without ap-
plying a Vietnamese word segmenter, those pre-
trained multilingual seq2seq models directly apply
Byte-Pair encoding models (Sennrich et al., 2016;
Kudo and Richardson, 2018) to the syllable-level
Vietnamese pre-training data. Therefore, it is inter-
esting to investigate the influence of word segmen-
tation on seq2seq pre-training for Vietnamese.
In this paper, we introduce BARTpho with two
versions—BARTphosyllable and BARTphoword —the
first large-scale monolingual seq2seq models pre-
trained for Vietnamese, which are based on the
seq2seq denoising autoencoder BART (Lewis et al.,
2020). The difference between our two BART-
pho versions is that they take different types of
input texts: a syllable level for BARTphosyllable
vs. a word level for BARTphoword . We compare
BARTpho with mBART (Liu et al., 2020)—a mul-
tilingual variant of BART—on a downstream task
1
Note that 85% of Vietnamese word types are composed
of at least two syllables (Thang et al., 2008).of Vietnamese text summarization. We find that
our BARTpho models outperform mBART in both
automatic and human evaluations, and help pro-
duce a new SOTA performance, thus showing the
effectiveness of large-scale monolingual seq2seq
pre-training for Vietnamese. We also find that
BARTphoword does better than BARTphosyllable ,
showing the positive influence of Vietnamese word
segmentation towards seq2seq pre-training.
We publicly release our BARTpho models that
can be used with popular libraries fairseq (Ott
et al., 2019) and transformers (Wolf et al.,
2020). We hope that our BARTpho can serve as a
strong baseline for future research and applications
of generative natural language processing (NLP)
tasks for Vietnamese.

## 2 Related work
PhoBERT (Nguyen and Nguyen, 2020) is the first
public large-scale monolingual language model
pre-trained for Vietnamese, which helps obtain
state-of-the-art performances on various down-
stream Vietnamese NLP/NLU tasks (Truong et al.,
2021; Nguyen and Nguyen, 2021; Dao et al., 2021;
Thin et al., 2021). PhoBERT is pre-trained on a
20GB word-level corpus of Vietnamese texts, us-
ing the RoBERTa pre-training approach (Liu et al.,
2019) that optimizes BERT for more robust perfor-
mance. Following PhoBERT, there are also public
monolingual language models for Vietnamese such
as viBERT and vELECTRA (Bui et al., 2020),
which are based on BERT and ELECTRA pre-
training approaches (Devlin et al., 2019; Clark
et al., 2020) and pre-trained on syllable-level Viet-
namese text corpora. Following Rothe et al. (2020)
who leverage pre-trained language model check-
points for sequence generation tasks, Nguyen et al.
(2021) conduct an empirical study and show that
PhoBERT helps produce better performance results
than viBERT for a downstream task of Vietnamese
abstractive summarization.
Our BARTpho is based on BART. We employ
BART because it helps produce the strongest per-
formances on downstream tasks in comparison to
other pre-trained seq2seq models under a compara-
ble setting in terms of the relatively equal numbers
of model parameters and pre-training data sizes
(Lewis et al., 2020; Raffel et al., 2020; Qi et al.,
2020). BART is also used to pre-train monolingual
models for other languages such as French (Eddine
et al., 2020) and Chinese (Shao et al., 2021).

## 3 Our BARTpho
This section describes the architecture, the pre-
training data and the optimization setup, that we
use for BARTpho.

### 3.1 Architecture

Both BARTphosyllable and BARTphoword use the
“large” architecture with 12 encoder and decoder
layers and pre-training scheme of BART (Lewis
et al., 2020). In particular, pre-training BART has
two stages: (i) corrupting the input text with an
arbitrary noising function, and (ii) learning to re-
construct the original text, i.e. optimizing the cross-
entropy between its decoder’s output and the origi-
nal text. Here, BART uses the standard architecture
Transformer (Vaswani et al., 2017), but employ-
ing the GeLU activation function (Hendrycks and
Gimpel, 2016) rather than ReLU and performing
parameter initialization from N (0, 0.02). Follow-
ing Liu et al. (2020), we add a layer-normalization
layer on top of both the encoder and decoder. Fol-
lowing Lewis et al. (2020), we also employ two
types of noise in the noising function, including
text infilling and sentence permutation. For text
infilling, we sample a number of text spans with
their lengths drawn from a Poisson distribution (λ
= 3.5) and replace each span with a single special
<mask> token. For sentence permutation, consec-
utive sentences are grouped to generate sentence
blocks of 512 tokens, and sentences in each block
are then shuffled in random order.

### 3.2 Pre-training data

For BARTphoword , we employ the PhoBERT pre-
training corpus (Nguyen and Nguyen, 2020), that
contains 20GB of uncompressed texts (about 145M
automatically word-segmented sentences). In ad-
dition, we also reuse the PhoBERT’s tokenizer
that applies a vocabulary of 64K subword types
and BPE (Sennrich et al., 2016) to segment those
word-segmented sentences with subword units.
BARTphoword has about 420M parameters. Pre-
training data for BARTphosyllable is a detokenized
variant of the PhoBERT pre-training corpus (i.e.
about 4B syllable tokens). We employ the pre-
trained SentencePiece model (Kudo and Richard-
son, 2018) from XLM-RoBERTa (Conneau et al.,
2020), used in mBART (Liu et al., 2020), to seg-
ment sentences with sub-syllable units and select
a vocabulary of the top 40K most frequent types.
BARTphosyllable has about 396M parameters.Model
mBART
BARTphosyllable
BARTphoword
Validation set
R-1
R-2
R-L
60.06 28.69 38.85
60.29 29.07 39.02
60.55 29.89 39.73
R-1
60.03
60.41
60.51
Test set
R-2
R-L
28.51 38.74
29.20 39.22
29.65 39.75
Human
21/100
37/100
42/100
Table 1: Detokenized and case-sensitive ROUGE scores (in %) w.r.t. duplicate article removal. R-1, R-2 and R-L
abbreviate ROUGE-1, ROUGE-2 and ROUGE-L, respectively. Every score difference between mBART and each
BARTpho version is statistically significant with p-value < 0.05.

### 3.3 Optimization

We utilize the BART implementation with the de-
noising task from fairseq (Ott et al., 2019). We
use Adam (Kingma and Ba, 2015) for optimization,
and use a batch size of 512 sequence blocks across
8 A100 GPUs (40GB each) and a peak learning
rate of 0.0001. Note that we initialize parameter
weights of BARTphosyllable by those from mBART.
For each BARTpho model, we run for 15 training
epochs in about 6 days (here, the learning rate is
warmed up for 1.5 epochs).

## 4 Experiments

### 4.1 Experimental setup

We evaluate and compare the performance of
BARTpho with the strong baseline mBART on
a downstream generative task of Vietnamese text
summarization. Here, mBART is pre-trained on
a Common Crawl dataset of 25 languages, which
includes 137 GB of syllable-level Vietnamese texts.
We employ the single-document summarization
dataset VNDS (Nguyen et al., 2019), consisting of
150704 news articles each including a news ab-
stract (i.e. gold summary) and body content (i.e.
input text). In particular, 105418, 22642 and 22644
articles are used for training, validation and test,
respectively. However, we find that there are dupli-
cate articles in this dataset. Therefore, we filter the
duplicates, resulting in 99134, 22184 and 22498 ar-
ticles for training, validation and test, respectively.2
When fine-tuning BARTphosyllable and mBART, we
use a detokenized version of the filtered dataset,
while its automatically word-segmented version is
used for fine-tuning BARTphoword .
We formulate this task as a monolingual transla-
tion problem and fine-tune our BARTpho and the
2
Firstly, we remove duplicates inside each of the training,
validation and test sets. Secondly, if an article appears in both
training and validation/test sets, then the article is filtered
out of the training set. Lastly, if an article appears in both
validation and test sets, then the article is filtered out of the
validation set.
baseline mBART using the same hyper-parameter
tuning strategy. We fix the maximum number of
tokens in a batch at 4096. We use Adam and run
for 20 training epochs. We also perform grid search
to select the Adam initial learning rate from {1e-5,
2e-5, 3e-5, 5e-5}. We employ beam search with
a beam size of 4 for decoding. We evaluate each
model 4 times in every epoch. We select the model
checkpoint that produces the highest ROUGE-L
score (Lin, 2004) on the validation set, and we then
apply the selected one to the test set.
Note that we compute the detokenized and case-
sensitive ROUGE scores for all models (here, we
detokenize the fine-tuned BARTphoword ’s output
before computing the scores).

### 4.2 Main results
Table 1 presents our obtained ROUGE scores on
the validation and test sets for the baseline mBART
and our two BARTpho versions w.r.t. the setting of
duplicate article removal. Clearly, both BARTpho
versions achieve significantly better ROUGE scores
than mBART on both validation and test sets.
We also conduct a human-based manual compar-
ison between the outputs produced by the baseline
mBART and our two BARTpho versions. In partic-
ular, we randomly sample 100 input text examples
from the test set; and for each input example, we
anonymously shuffle the summary outputs from
three fine-tuned models (here, each input sampled
example satisfies that any two out of three sum-
mary outputs are not exactly the same). We then
ask two external Vietnamese annotators to choose
which summary they think is the best. We obtain
a Cohen’s kappa coefficient at 0.61 for the inter-
annotator agreement between the two annotators.
Our second co-author then hosts and participates in
a discussion session with the two annotators to re-
solve annotation conflicts (here, he does not know
which model produces which summary). Table 1
shows final scores where BARTpho obtains a better
human evaluation result than mBART.from transformers import AutoModel, AutoTokenizer

* BARTphosyllable
tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
bartpho_syllable = AutoModel.from_pretrained("vinai/bartpho-syllable")
input_text = 'Chúng tôi là những nghiên cứu viên.'
input_ids = tokenizer(input_text, return_tensors='pt')
features = bartpho_syllable(∗∗input_ids)

* BARTphoword
tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word")
bartpho_word = AutoModel.from_pretrained("vinai/bartpho-word")
input_text = 'Chúng_tôi là những nghiên_cứu_viên .'
input_ids = tokenizer(input_text, return_tensors='pt')
features = bartpho_word(∗∗input_ids)
Figure 1: An example code using BARTpho for feature extraction with transformers in Python.
Model
fastAbs [?]
viBERT2viBERT [∗]
PhoBERT2PhoBERT [∗]
mT5 [∗]
mBART
BARTphosyllable
BARTphoword
Original test set
R-1
R-2
R-L
54.52 23.01 37.64
59.75 27.29 36.79
60.37 29.12 39.44
58.05 26.76 37.38
60.35 29.13 39.21
60.88 29.90 39.64
61.14 30.31 40.15
Table 2: ROUGE scores (in %) w.r.t. the original
dataset setting (i.e. without duplicate article removal).
[?] denotes the best performing model among different
models experimented by Nguyen et al. (2019). [∗] de-
notes scores reported by Nguyen et al. (2021).
For comparison with previously published re-
sults (Nguyen et al., 2019, 2021), we also fine-tune
our BARTpho models and baseline mBART on
the original training set (i.e. without duplicate ar-
ticle removal),3 using the same hyper-parameter
tuning strategy as presented in Section 4.1. We
report ROUGE scores on the original test set in
Table 2. The previous highest ROUGE-L score
obtained among different models experimented by
Nguyen et al. (2019, 2021) is 39.44 accounted for
PhoBERT2PhoBERT, that is 0.2 and 0.7 points
lower than BARTphosyllable and BARTphoword , re-
spectively. Tables 1 and 2 show that BARTpho
helps attain a new SOTA performance for this task.
Our automatic and human evaluation results
from tables 1 and 2 demonstrate the effectiveness
of large-scale BART-based monolingual seq2seq
models for Vietnamese. Note that mBART uses
137 / 20 ≈ 7 times bigger Vietnamese pre-training
data than BARTpho. In addition, the multilingual
3
This is not a proper experimental setup because of data
leakage, e.g. 1466 training articles appear in the test set.
seq2seq mT5 (Xue et al., 2021b) is pre-trained on
the multilingual dataset mC4 that includes 79M
Common Crawl Vietnamese pages consisting of
116B syllable tokens, i.e. mT5 uses 116 / 4 =
29 times bigger Vietnamese pre-training data than
BARTpho. However, BARTpho surpasses both
mBART and mT5, reconfirming that the dedicated
language-specific model still performs better than
the multilingual one (Nguyen and Nguyen, 2020).
Tables 1 and 2 also show that BARTphoword outper-
forms BARTphosyllable , thus demonstrating the pos-
itive influence of word segmentation for seq2seq
pre-training and fine-tuning in Vietnamese.

## 5 Usage example
Figure 1 presents a basic usage of BARTpho for
feature extraction with transformers to show
its potential use for other downstream tasks.4 More
usage examples of BARTpho with both fairseq
and transformers can be found at the BART-
pho’s GitHub repository: https://github.com/VinAIResearch/BARTpho

## 6 Conclusion
In this paper, we have presented BARTphosyllable
and BARTphoword —the first pre-trained and large-
scale monolingual seq2seq models for Vietnamese.
We demonstrate the usefulness of our BARTpho
by showing that BARTpho performs better than its
competitor mBART and helps produce the SOTA
performance for the Vietnamese text summariza-
tion task. We hope that our public BARTpho mod-
els can foster future research and applications of
generative Vietnamese NLP tasks.