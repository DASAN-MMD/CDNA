# CDNA

Wi-Fi sensor networks have grown rapidly due to their scalability and high
data throughput, finding applications in tasks like tracking human motion in
laboratory settings. These systems detect motion by analyzing fluctuations
in radio signals caused by target movements, which generate identifiable
activity patterns. However, their performance is influenced by factors such
as environmental changes, unseen target subjects, multi-target tracking, data
configurations, and the nature of target activities. These challenges lead to
domain shifts between training and testing phases, a common issue in real-world
scenarios known as the domain-shifting problem in transfer learning. We propose
a supervised domain alignment technique to address domain shifts in Wi-Fi sensor
Channel State Information (CSI) datasets using minimal labeled target data.
Our method outperforms state-of-the-art adversarial models trained on similar
data, achieving superior cross-domain prediction accuracy. Evaluations on two
public CSI datasets show consistent improvements, with an average Micro-F1
score of 90% for cross-user tasks and 67% for cross-user and cross-environment
tasks using only 70 labeled target samples. This approach demonstrates its
effectiveness in enhancing prediction accuracy under challenging domain shift
scenario.
