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
from streams.readers.arff_reader import ARFFReader
from tasks.__init__ import *


# 1. Creating a project
project = Project("projects/single", "lr-mit")

# 2. Loading an arff file
labels, attributes, stream_records = ARFFReader.read("arff-dataset/final.arff")
attributes_scheme = AttributeScheme.get_scheme(attributes)

# 3. Initializing a Learner
learner = LogisticRegression(labels, attributes_scheme["nominal"])

# 4. Initializing a drift detector
detector = FHDDM(n=100)
actual_drift_points = [
    200,
    500,
    600,
    900,
    1200,
    1500,
    1800,
    2000,
    2300,
    2600,
    2700,
    3000,
    3250,
    3500,
    3800,
    4000,
    4300,
    4800,
    5000,
    5300,
    5600,
    6000,
    6150,
    6500,
    6800,
    7000,
    7350,
    7600,
    7900,
    8200,
    8500,
    8800,
    9000,
    9400,
    9700,
    10000,
    10250,
    10500,
    10850,
    11150,
    11450,
    11700,
]
drift_acceptance_interval = 250

# 5. Creating a Prequential Evaluation Process
prequential = PrequentialDriftEvaluator(
    learner,
    detector,
    attributes,
    attributes_scheme,
    actual_drift_points,
    drift_acceptance_interval,
    project,
)

# prequential = PrequentialDrift(learner, detector, attributes, attributes_scheme, project)

# prequential = Prequential(learner, attributes, attributes_scheme, project)

prequential.run(stream_records, 1)
