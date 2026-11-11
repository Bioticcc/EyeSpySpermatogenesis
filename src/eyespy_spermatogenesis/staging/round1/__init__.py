"""Staging round 1 (decision 023): frozen features + small classifier.

Labels: Yeongseok's paper stages from the annotation workspace. Steps:
``data`` (label table + native crops), ``embed`` (frozen feature extractor on
224 px patches), ``evaluate`` (leave-one-section-out classifiers). The same
code serves round 3 (UNI2) with a different extractor.
"""
