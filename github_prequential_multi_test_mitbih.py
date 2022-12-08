"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""

from data_structures.attribute_scheme import AttributeScheme
from classifier.__init__ import *
from drift_detection.__init__ import *
from filters.project_creator import Project
from graphic.hex_colors import Color
from streams.readers.arff_reader import ARFFReader
from tasks.prequential_learner_detector_pairs import PrequentialMultiPairs

# 1. Creating a project
project = Project("projects/multi", "sine")

# 2. Loading an arff file
labels, attributes, stream_records = ARFFReader.read(
    "data_streams/sine1_w_50_n_0.1/sine1_w_50_n_0.1_101.arff"
)
attributes_scheme = AttributeScheme.get_scheme(attributes)

# 3. Initializing a Classifier-Detector Pairs
pairs = [
    [NaiveBayes(labels, attributes_scheme["nominal"]), FHDDM()],
    [NaiveBayes(labels, attributes_scheme["nominal"]), FHDDMS()],
    [NaiveBayes(labels, attributes_scheme["nominal"]), CUSUM()],
    [NaiveBayes(labels, attributes_scheme["nominal"]), PH()],
    [NaiveBayes(labels, attributes_scheme["nominal"]), DDM()],
    [NaiveBayes(labels, attributes_scheme["nominal"]), EDDM()],
    [NaiveBayes(labels, attributes_scheme["nominal"]), ADWINChangeDetector()],
    [NaiveBayes(labels, attributes_scheme["nominal"]), SeqDrift2ChangeDetector()],
    [NaiveBayes(labels, attributes_scheme["nominal"]), HDDM_A_test()],
    [NaiveBayes(labels, attributes_scheme["nominal"]), HDDM_W_test()],
    [LogisticRegression(labels, attributes_scheme["nominal"]), FHDDM()],
    [LogisticRegression(labels, attributes_scheme["nominal"]), FHDDMS()],
    [LogisticRegression(labels, attributes_scheme["nominal"]), CUSUM()],
    [LogisticRegression(labels, attributes_scheme["nominal"]), PH()],
    [LogisticRegression(labels, attributes_scheme["nominal"]), DDM()],
    [LogisticRegression(labels, attributes_scheme["nominal"]), EDDM()],
    [LogisticRegression(labels, attributes_scheme["nominal"]), ADWINChangeDetector()],
    [
        LogisticRegression(labels, attributes_scheme["nominal"]),
        SeqDrift2ChangeDetector(),
    ],
    [LogisticRegression(labels, attributes_scheme["nominal"]), HDDM_A_test()],
    [LogisticRegression(labels, attributes_scheme["nominal"]), HDDM_W_test()],
    [Perceptron(labels, attributes_scheme["numeric"]), FHDDM()],
    [Perceptron(labels, attributes_scheme["numeric"]), FHDDMS()],
    [Perceptron(labels, attributes_scheme["numeric"]), CUSUM()],
    [Perceptron(labels, attributes_scheme["numeric"]), PH()],
    [Perceptron(labels, attributes_scheme["numeric"]), DDM()],
    [Perceptron(labels, attributes_scheme["numeric"]), EDDM()],
    [Perceptron(labels, attributes_scheme["numeric"]), ADWINChangeDetector()],
    [Perceptron(labels, attributes_scheme["numeric"]), SeqDrift2ChangeDetector()],
    [Perceptron(labels, attributes_scheme["numeric"]), HDDM_A_test()],
    [Perceptron(labels, attributes_scheme["numeric"]), HDDM_W_test()],
    [LogLinear(labels, attributes_scheme["nominal"]), FHDDM()],
    [LogLinear(labels, attributes_scheme["nominal"]), FHDDMS()],
    [LogLinear(labels, attributes_scheme["nominal"]), CUSUM()],
    [LogLinear(labels, attributes_scheme["nominal"]), PH()],
    [LogLinear(labels, attributes_scheme["nominal"]), DDM()],
    [LogLinear(labels, attributes_scheme["nominal"]), EDDM()],
    [LogLinear(labels, attributes_scheme["nominal"]), ADWINChangeDetector()],
    [
        LogLinear(labels, attributes_scheme["nominal"]),
        SeqDrift2ChangeDetector(),
    ],
    [LogLinear(labels, attributes_scheme["nominal"]), HDDM_A_test()],
    [LogLinear(labels, attributes_scheme["nominal"]), HDDM_W_test()],
    [HoeffdingTree(labels, attributes_scheme["nominal"]), FHDDM()],
    [HoeffdingTree(labels, attributes_scheme["nominal"]), FHDDMS()],
    [HoeffdingTree(labels, attributes_scheme["nominal"]), CUSUM()],
    [HoeffdingTree(labels, attributes_scheme["nominal"]), PH()],
    [HoeffdingTree(labels, attributes_scheme["nominal"]), DDM()],
    [HoeffdingTree(labels, attributes_scheme["nominal"]), EDDM()],
    [HoeffdingTree(labels, attributes_scheme["nominal"]), ADWINChangeDetector()],
    [HoeffdingTree(labels, attributes_scheme["nominal"]), SeqDrift2ChangeDetector()],
    [HoeffdingTree(labels, attributes_scheme["nominal"]), HDDM_A_test()],
    [HoeffdingTree(labels, attributes_scheme["nominal"]), HDDM_W_test()],
]

# 4. Creating a color set for plotting results
colors = [
    Color.Indigo[1],
    Color.Blue[1],
    Color.Green[1],
    Color.Lime[1],
    Color.Yellow[1],
    Color.Amber[1],
    Color.Orange[1],
    Color.Red[1],
    Color.Purple[1],
    Color.Pink[1],
    Color.Indigo[2],
    Color.Blue[2],
    Color.Green[2],
    Color.Lime[2],
    Color.Yellow[2],
    Color.Amber[2],
    Color.Orange[2],
    Color.Red[2],
    Color.Purple[2],
    Color.Pink[2],
    Color.Indigo[3],
    Color.Blue[3],
    Color.Green[3],
    Color.Lime[3],
    Color.Yellow[3],
    Color.Amber[3],
    Color.Orange[3],
    Color.Red[3],
    Color.Purple[3],
    Color.Pink[3],
    Color.Indigo[4],
    Color.Blue[4],
    Color.Green[4],
    Color.Lime[4],
    Color.Yellow[4],
    Color.Amber[4],
    Color.Orange[4],
    Color.Red[4],
    Color.Purple[4],
    Color.Pink[4],
    Color.Indigo[0],
    Color.Blue[0],
    Color.Green[0],
    Color.Lime[0],
    Color.Yellow[0],
    Color.Amber[0],
    Color.Orange[0],
    Color.Red[0],
    Color.Purple[0],
    Color.Pink[0],
]

# 5. Defining actual locations of drifts, acceptance delay interval, and vector of weights
actual_drift_points = [20000, 40000, 60000, 80000]
drift_acceptance_interval = 250
w_vec = [1, 1, 1, 1, 1, 1]

# 6. Creating a Prequential Evaluation Process
prequential = PrequentialMultiPairs(
    pairs,
    attributes,
    attributes_scheme,
    actual_drift_points,
    drift_acceptance_interval,
    w_vec,
    project,
    color_set=colors,
    legend_param=False,
)

prequential.run(stream_records, 1)
