# CTMNet (CTNet with Mixture of Experts) for EEG-based Motor Imagery Classification tasks

This work is based on CTNet, and we thank the CTNet authors for the open source project

CTNet: A Convolutional Transformer Network for EEG-Based Motor Imagery Classification

core idea: CNN (an improved version of EEGNet) + Transformer encoder with Mixture of Experts

# Abstract:
Brain-computer interfaces (BCIs) hold significant potential in various fields such as rehabilitation and pain alleviation by enabling machine control through EEG-based motor imagery (MI). However, individual variabilities and abnormalities in EEG signals restrict decoding performance and limit broader BCI applications. Recently, the incorporation of Mixture of Experts (MoE) into transformer architectures in the natural language processing domain has demonstrated improved performance and computational efficiency. In this paper, we propose the use of a convolutional transformer with MoE network (CTMNet). The CTMNet consists of convolutional modules that are used to extracti local spatial-temporal features from EEG signals and integrates a MoE into the transformer encoder to efficiently capture global dependencies. Evaluated on the BCI Competition IV-2a and IV-2b datasets, the CTMNet achieved subject-specific classification accuracies of 86.6% and 90.72%. In cross-subject evaluations, it reached accuracies of 63.39% and 78.52%. These results indicate that this architecture can effectively enhance MI-EEG classification performance.


Citation : K. Minhyeok and A. Corradini, ‘Combining Mixture of Experts with Transformer for EEG-based Motor Imagery Classification’, in 2025 International Conference on Artificial Intelligence, Computer, Data Sciences and Applications (ACDSA), 2025, pp. 1–6.

Paper : 10.1109/ACDSA65407.2025.11166255

Email : M.Kim23@liverpool.ac.uk / mh20151231@gmail.com
