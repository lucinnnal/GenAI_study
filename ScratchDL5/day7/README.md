### Day 7 - Conditional Diffusion & Guidance

On **Day 7**, we covered the concept of **Conditional Diffusion Models**, which are designed to generate data conditioned on given inputs (e.g., class labels, text descriptions).

Additionally, we explored **Guidance** techniques that allow us to further emphasize the conditioning signal during generation. Specifically, for **score-based diffusion models**, we studied two major approaches:

- **Classifier Guidance**: Uses an external classifier to guide the generation process toward the desired condition.
- **Classifier-Free Guidance (CFG)**: A unified model trained with both conditional and unconditional inputs, allowing conditional emphasis through interpolation at sampling time.

During the coding session, we implemented and trained a **Classifier-Free Guidance** model. The hands-on practice helped us understand how the interpolation of conditional and unconditional predictions enables controllable generation.

- 📄 The **Main Note** is documented in ![note](./day6_&_day7.pdf).
- 📁 For supplementary materials, refer to the `supplements/` directory.