## Aim

owNet is the net of Unit 4, "Build your own": a small transformer encoder–decoder that learns to distribute ⨂ over ⨁. Given the tokens of `(a⨁b)⨂c`, it writes `⨁ ⨂ a c ⨂ b c`, the distributed form in prefix order.

## Where it stands

It was trained in Wolfram in Unit 4, and the trained net is kept in `unit4_build_your_own_nets.nb` (as `owNet = Uncompress[…]`), where it is restored and checked in a fresh process. It has not yet been trained in the lab's form, so there is no recorded run here yet. The diagram is in the net-diagram editor, which writes its training notebook (`owNetFromDiagram.nb`); a generated owNet has already been trained on a copy task as a check.

## Next

Record a run in the lab's form, then a 3D view for sequence nets.
