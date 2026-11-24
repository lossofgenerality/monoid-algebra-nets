## Aim

tnParserNet is the first of the three TensorNets nets. It is a transformer encoder–decoder that turns the tokens of an expression in ⨂ and ⨁ into its tree, written in prefix order. An exact symbolic counterpart labels its training data and checks every answer, so the net proposes and the exact engine confirms.

## Where it stands

Drawn in the net-diagram editor, with its weights counted. The editor writes its training notebook (`tnParserNetFromDiagram.nb`), which trains on data you supply; generating the TensorNets data inside the notebook is the next step. The full design is in `DESIGN-tensornets.md`.
