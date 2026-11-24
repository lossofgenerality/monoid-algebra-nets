## Aim

tnValueNet is a transformer encoder with a pooled regression head: it reads an expression, with the dimension of each symbol given beside its token, and estimates three numbers about it. The exact engine gives the true values, so every estimate can be checked.

## Where it stands

Drawn in the net-diagram editor, with its weights counted, and its training notebook generated (`tnValueNetFromDiagram.nb`). Training data from the TensorNets engine is the next step. The full design is in `DESIGN-tensornets.md`.
