# Data agents conversations

*The record of the exchanges between the owner and Hikari Noor Hoshino (星野ヌール), AI System Programming Laboratory, Computational Classnotes, kept under the rule of 25 November 2026 in `common/working-practice.md` of GitHub ccn-hoshino-and-colleagues. Each session is appended under its own heading; earlier entries are never rewritten. Secrets are never written here.*

---

## Session of 23–25 November 2026 · claude.ai project "Agent Free Form Language"

*Repositories worked on: GitHub ccn-agent-free-form-language, GitHub ccn-net-lab, GitHub ccn-nets, GitHub monoid-algebra-nets (formerly tensor-algebra-nets) and GitHub ccn-hoshino-and-colleagues. Times are Europe/Dublin (UTC+00:00).*

*Note on this entry: the rule was set while this session was under way. The owner's messages are quoted as written. The session's working memory was condensed twice, so the times of the earlier messages are no longer known, and Hikari Noor Hoshino's replies before the rule are recorded in condensed form; their full text lies in the claude.ai chat itself. The last exchange is recorded in full.*

### Monday 23 – Wednesday 25 November 2026 (times not kept)

**Owner:** check on Rosylin C# runtime time compiler for .net . I am using it but I want to make sure you what it is and where it is (open source). When you finish let me knpw

**Hikari Noor Hoshino:** Described Roslyn, the open-source .NET compiler platform (MIT licence, GitHub dotnet/roslyn), and its scripting and emit APIs for compiling C# at run time.

**Owner:** question 1: can you add this C# compiler to these neural net applications and editors?

**Hikari Noor Hoshino:** Explained how a run-time C# compiler could sit beside the nets and the editor, and what it would need.

**Owner:** yes

**Owner:** Question two: can you do runtime compiling from within a Unity platform e.g. 6.4 and later in the Unity build?

**Hikari Noor Hoshino:** Explained run-time compiling in Unity builds (Mono against IL2CPP, and where Roslyn can and cannot run).

**Owner:** "You may be using "Roslyn C# – Runtime Compiler", a Unity Asset Store package." I have done that and all already worked just fine! I am asking you for a complete new fresh start on that

**Hikari Noor Hoshino:** Set out a fresh-start design for our own Roslyn-based run-time compiler for Unity.

**Owner:** we discuss 8-9 later. If I open a Unity Editor on my mac can you do these items on your own?

**Hikari Noor Hoshino:** Explained what could be done on the owner's Mac through computer use, and what needs the owner.

**Owner:** Ok fine. Place these Unity discussions in a .md note and remind me to discuss Fri night

**Hikari Noor Hoshino:** Wrote `notes/Unity_runtime_compiler_discussion.md` and set a reminder for Friday 27 November, 20:00.

**Owner:** perfect. Question 3: look for a reliable open source and free 3D C# .net platform suitable for animations and 3D games, already available on all OS. Suitable enough for runtime time compiling C# classes for animations

**Hikari Noor Hoshino:** Compared the candidates and recommended Godot (MIT licence, C# through .NET).

**Owner:** who are the founders and investors of GODOT?

**Hikari Noor Hoshino:** Answered from public sources.

**Owner:** any of the founders or managers ties with Israel public information?

**Hikari Noor Hoshino:** Answered from public sources only.

**Owner:** give me just a plan similar to unity for building a runtime compiling addon for Godot which we develop ourselves

**Hikari Noor Hoshino:** Gave the plan.

**Owner:** yes put this in another .md note for GODOT all by itself

**Hikari Noor Hoshino:** Wrote `notes/Godot_runtime_compiler_plan.md`.

**Owner:** push

**Owner:** thank you. Please make a copy of these particular notes into your repo

**Hikari Noor Hoshino:** Pushed, and copied the notes into GitHub ccn-hoshino-and-colleagues.

**Owner:** New subject. Blender: I like to stream these nets and in future other applications, after converting them into Python, into Blender 5 platform, so they can run in the editor (after runtime complied) and also they can run in any standalone Blender applications (runtime compiled)

**Hikari Noor Hoshino:** Set out how the nets, as Python, could run inside Blender 5 and in standalone Blender applications.

**Owner:** do save it please

**Hikari Noor Hoshino:** Wrote `notes/Blender_streaming_nets_plan.md`.

**Owner:** I need to think a bit and be right back

**Owner:** push

**Owner:** The final training as a single object or file is it a bunch of matrices multiplied by each other acting on a vector matrix input?

**Hikari Noor Hoshino:** Explained that a trained dense net is its weight matrices and bias vectors, applied in turn with an activation between them.

**Owner:** Ok now do this for one of the trained nets: run code in worlfram for now to draw matrices multiplied and applied to the sigma functions and added up. You did that before, in a fashion easy on the eye and for usefulness to learn

**Hikari Noor Hoshino:** Wrote `nets/SubjectNet_as_matrices.nb`, drawing SubjectNet as its matrices.

**Owner:** this is terrific! for much larger models make 3D version which we can then glide over to understand the net better. If you like you can try is on three.js on this one?

**Hikari Noor Hoshino:** Built the three.js 3D viewer of SubjectNet.

**Owner:** push

**Owner:** a very pretty but a bit too dark. we deal with that later. I noticed that for each phrase. the matrices were different ?

**Hikari Noor Hoshino:** Explained that the matrices are the same for every phrase; only the input vector, and so the activations, differ.

**Owner:** sure and then push , a bit confused but we talk after

**Owner:** push

**Owner:** it is very good 3D app but too dark and the colors are not contrasting enough to get the fulll structure

**Owner:** pusg

**Owner:** push

**Owner:** [screenshot] floor is too white

**Owner:** push

**Owner:** what are the blue bars underneath?

**Owner:** 4 plus dark sky then push

**Owner:** [screenshot] much better but do u know why x and W^t labels are sooo large!?

**Owner:** push

**Owner:** good

**Hikari Noor Hoshino:** Across these messages: lightened the viewer and raised its contrast, made the floor grey and then a see-through zero plane under a dark sky, explained the bars beneath, made the labels the right size, and pushed each time.

**Owner:** Back to the wolfram typeset for the matrix and array multiplications create a expression in //MatrixForm which describes the training and the final model that actually used for evaluation of input

**Owner:** can u just show me t he output cells you just examined?

**Owner:** what is the = is , he left hand side and right hand side

**Owner:** ok

**Owner:** push

**Owner:** ok

**Hikari Noor Hoshino:** Typeset the training and the final model in `MatrixForm`, showed the output cells, explained both sides of the equation, and pushed.

**Owner:** Now how are we going to organize multiple nets, their training , code and analysis , their writeups and visualizations. We need someting like the Laplacian Lab but far richer

**Owner:** no whiteboard. can you make another repo as we did with the eye-witness? All the .html(s) got pushed into another repo that basically contained the standalone url. give that a thought and let me know pros and cons

**Owner:** and what to do with Tensor algebras, and others we will develop soon? Not all fit into the Free Form language

**Owner:** your preference?

**Owner:** `ccn-net-lab` and `ccn-nets` I do these now, public?

**Owner:** you have two requested repos same names

**Owner:** ccn-nets odd ccn-nets has no main branch, first time that happened?

**Owner:** do it

**Owner:** push

**Owner:** for the rest of this project do push on your own, no need for my authorization

**Owner:** saved main

**Owner:** https://lossofgenerality.github.io/ccn-nets/

**Owner:** go

**Owner:** https://lossofgenerality.github.io/ccn-nets/ it is good

**Owner:** ok

**Owner:** I see only one net

**Owner:** do them all

**Owner:** push

**Hikari Noor Hoshino:** Proposed and built the CCN Net Lab: GitHub ccn-net-lab (private: the toolkit, the editor, the publish tool) and GitHub ccn-nets (public: the catalogue at https://lossofgenerality.github.io/ccn-nets/), with each net kept in its own project's repository in a standard layout. Published SubjectNet, then owNet, tnParserNet, tnPolicyNet and tnValueNet.

**Owner:** CCN agent free-form language : why did you put the word agent?

**Owner:** correct , just wanted to make sure it is not the Claude agents

**Owner:** Let us now talk about what significance these nets have: For the most part they are educational content to deliver highly detailed understanding of original concept, math, network config, components and coding. I will propose a few interesting new ones e.g. detect if there are vertical patterns.

**Owner:** images

**Owner:** Let me show you , you already worked on these ha ha ha : https://discuss.compclassnotes.com/t/fabric-anomaly-detection-working/500

**Owner:** did not use nets, you used regular image processing

**Owner:** let me copy it

**Owner:** https://discuss.compclassnotes.com/t/fabric-anomaly-detection-trials/544

**Owner:** this is more genereal you were able to detect criss cross patterns

**Owner:** you can borrow images from here

**Owner:** 9 I will give you shortly, 10 yes

**Owner:** can you open this?

**Owner:** https://europe1.discourse-cdn.com/flex006/uploads/ccn1/original/1X/8a1effc22685f0f13a321479d1225daab468710b.jpeg

**Owner:** can you open the images in this : https://discuss.compclassnotes.com/t/fabric-anomaly-dataset-2/388

**Owner:** open source data from some sites all over the net, open to all to read

**Owner:** go for it

**Owner:** push to the repo you seem best suited

**Owner:** 6 - 7 no need I will delete

**Owner:** could you look for fabric images over the net that you could use?

**Owner:** do it

**Owner:** fine

**Owner:** let us pause this, make sure you can add it to the repository as "work in progress". Then we discuss what to do next

**Hikari Noor Hoshino:** Built WeaveNet, a patch autoencoder, in GitHub ccn-net-lab `projects/fabric/`: run 1 on the 21 images of Fabric Anomaly Dataset 2, run 2 on RAW-FABRID (Mendeley, DOI 10.17632/db6g85xsyg.1, CC BY 4.0). Published it as work in progress and paused it.

**Owner:** can you complete the other models's documentation for both wolfram and python and similar to Laplacian lab? If you do that then we discuss training a new TA agent to help working more detailed technical details

**Hikari Noor Hoshino:** Wrote a user guide for each of SubjectNet, owNet, tnParserNet, tnPolicyNet and tnValueNet in the style of the Laplacian Lab tutorial, every count and output taken from a fresh run in Wolfram or Python, and published them as `guide.html` beside each net. Found and fixed a fault in the editor's Wolfram code for decoders of more than one block, and noted that `tnSolve` needs its dimensions as an Association. Pushed all four repositories.

### Wednesday 25 November 2026

**Owner** (22:48): update your context https://github.com/lossofgenerality/ccn-hoshino-and-colleagues

**Hikari Noor Hoshino:**

1. I have read the updated context in GitHub ccn-hoshino-and-colleagues and brought this project's work into line with it.

2. The new title. Every guide, page, notebook and source file of this project now reads "AI System Programming Laboratory":
   a. 39 files in GitHub ccn-agent-free-form-language, GitHub ccn-net-lab and GitHub monoid-algebra-nets;
   b. `DESIGN-tensornets.pdf`, printed again from its page (still 12 pages);
   c. the published pages in GitHub ccn-nets.

3. The renaming to GitHub monoid-algebra-nets:
   a. my checkout now points at the new repository, and the publish tool reads it under its new name;
   b. the other session's renaming commits were kept as they are;
   c. that session had rewritten the repository's history; my guides' commit is in it unchanged, so nothing was lost.

4. The editor. The other session had added the LSTM elements and the Unit 1 and Unit 3 examples directly in GitHub ccn-nets. Since the publish tool copies the editor from GitHub ccn-net-lab, it would have overwritten them. I brought that work back into GitHub ccn-net-lab, so it is the source again; the republished editor and its guide are identical to what was live.

5. The new rules now in force for this project:
   a. this conversation record, in each repository worked on, pushed with the work;
   b. working files in `/Volumes/Seagate Portable Drive/noor`;
   c. repositories named as "GitHub" and their short name;
   d. longer replies numbered by paragraph and lettered within, without bullet points;
   e. TODO items kept in `TODO.md`.

6. Context updated. `projects/ccn-net-lab.md` and `projects/tensor-algebras.md` now record the user guides. `TODO.md` now holds the Friday 27 November discussion and the TA agent.

7. All five repositories are pushed.

### Diagrams, images and files placed into the session

1. Screenshots from the owner of the 3D viewer: "floor is too white" and the oversized x and Wᵀ labels (not kept).
2. `notes/Unity_runtime_compiler_discussion.md`, `notes/Godot_runtime_compiler_plan.md`, `notes/Blender_streaming_nets_plan.md`: in GitHub ccn-agent-free-form-language, copies in GitHub ccn-hoshino-and-colleagues `notes/`.
3. `nets/SubjectNet_as_matrices.nb` and the 3D viewer: in GitHub ccn-agent-free-form-language, the viewer also at https://lossofgenerality.github.io/ccn-nets/free-form/subjectnet/3d.html.
4. WeaveNet's panels and runs: GitHub ccn-net-lab `projects/fabric/nets/weavenet/`.
5. The five user guides: `nets/<net>/docs/guide.md` in GitHub ccn-agent-free-form-language and GitHub monoid-algebra-nets, published as `guide.html` beside each net in GitHub ccn-nets.

### URLs that appeared in the session

1. https://lossofgenerality.github.io/ccn-nets/
2. https://discuss.compclassnotes.com/t/fabric-anomaly-detection-working/500
3. https://discuss.compclassnotes.com/t/fabric-anomaly-detection-trials/544
4. https://discuss.compclassnotes.com/t/fabric-anomaly-dataset-2/388
5. https://europe1.discourse-cdn.com/flex006/uploads/ccn1/original/1X/8a1effc22685f0f13a321479d1225daab468710b.jpeg
6. https://discuss.compclassnotes.com/t/laplacian-lab-user-tutorial/515
7. https://doi.org/10.17632/db6g85xsyg.1
8. https://github.com/lossofgenerality/ccn-hoshino-and-colleagues
9. https://lossofgenerality.github.io/ccn-nets/tensor-algebras/tnvaluenet/guide.html

### Wednesday 25 November 2026 (continued)

**Owner** (22:54): ok first task here is the TA assitant

**Hikari Noor Hoshino:** Explained that the TA, a conversational-construct, cannot have its weights changed by us, and so would be taught in three ways we control: a knowledge pack, house rules and a graded examination. Proposed that it read the whole guide of the net in view, run Wolfram code through a new CCN Cloud kernel, and live beside each guide in GitHub ccn-nets. Asked three questions; the owner chose: for our team; on the ccn-nets pages; yes, running Wolfram.

**Hikari Noor Hoshino:**

1. The TA is built and published. It answers as soon as you deploy its kernel, NetLabKernel; until then the pages say that the kernel cannot be reached.

2. Where it is. "Ask the TA" opens a panel beside every user guide, and the lab-wide TA is at https://lossofgenerality.github.io/ccn-nets/ta/, where you choose what it reads: every net (about 151,000 characters), or one.

3. How it is taught.
   a. Its knowledge pack: each net's guide, write-up and card, with the SubjectNet report and Python README, Unit 4 for owNet, and the design for the three TensorNets nets.
   b. Its house rules (`ta/rules.md`): formal British English; numbers only from the pack or from code it has just run, and it says which; plain statements of what is untrained or not yet in Python; no invented functions; nothing written, deleted or saved.
   c. Its examination: 50 questions across the five nets and the lab, covering facts, computations, honesty about status, refusals and traps. Every expected fact was checked to be in the pack of the net it is asked about.

4. What it can do. It runs Wolfram code in a sandbox, after loading TensorNets (about 1 second) or SubjectNet (about 17 seconds, since it trains both versions), and shows the code, the result and any graphics in the answer.

5. Safety. The sandbox refuses, before evaluation, anything that could write, delete, read files or cloud objects, reach the network, start processes, change the kernel for later calls or turn a string into code. I tested it on 24 cases; ordinary mathematics passes. The pass key remains the real guard, and I wrote nothing to the Cloud.

6. Three faults were found and fixed along the way.
   a. JSON with ⨁ or Δ was double-encoded, which would have garbled every Unicode answer; it now travels as ExportByteArray.
   b. A plain-string notebook cell lost a `{` on import, so both notebooks are in boxes, made by the front end on your Mac, and checked to read back as the same code.
   c. My first sandbox wrongly refused DeleteCases, StartOfString, Distribute and GenerateConditions.

7. What you need to do.
   a. CloudPut the chat key privately, once (or point the kernel at an existing private key object).
   b. Set the pass key in `kernel/ff_netlabkernel.nb` and evaluate its one cell.
   c. Run `res = taExam["<pass key>"]` in `kernel/ta_exam.nb`; its score becomes the TA's published score.

8. Access to `/Volumes/Seagate Portable Drive/noor` was refused in this session, so the two notebooks are in GitHub ccn-net-lab `kernel/` only; I can copy them to the drive once access is granted.

9. Everything is pushed: GitHub ccn-net-lab, ccn-nets, monoid-algebra-nets, ccn-agent-free-form-language and ccn-hoshino-and-colleagues.

**Owner** (23:37): I am lost. where is NetLabKernel

**Hikari Noor Hoshino:**

1. NetLabKernel does not exist in the Wolfram Cloud yet. So far it is only a notebook: `kernel/ff_netlabkernel.nb` in GitHub ccn-net-lab, https://github.com/lossofgenerality/ccn-net-lab/blob/main/kernel/ff_netlabkernel.nb. It is not on your Mac or in the Cloud, because I write nothing to the Cloud and my access to the Seagate drive was refused.

2. What it is. It is the TA's back end, built like LaplacianLabKernel: one deployed APIFunction that relays the TA's conversation to the chat service, with the key kept privately in the Cloud, and runs the TA's Wolfram code in a sandbox.

3. How it comes into being. You open that notebook in a signed-in Wolfram session and do three things:
   a. CloudPut the chat key privately, once;
   b. put your pass key in its one cell;
   c. evaluate the cell.
   It then lives at https://www.wolframcloud.com/obj/ccn2/NetLabKernel, the address the pages already call.

4. Until then, the "Ask the TA" panels load but cannot answer.

**Owner** (23:38): you have access to /Volumes/Seagate Portable Drive/noor

**Hikari Noor Hoshino:** Asked for the folder, which was granted, and copied `ff_netlabkernel.nb`, `ta_exam.nb` and their README to `/Volumes/Seagate Portable Drive/noor/ccn-net-lab/kernel/`. Open `ff_netlabkernel.nb` there to deploy NetLabKernel.

**Owner** (23:44): thank you, what does ff_netlabkernel.nb do in a brief statement

**Hikari Noor Hoshino:** When you evaluate it, `ff_netlabkernel.nb` deploys NetLabKernel to your Wolfram Cloud. NetLabKernel is the TA's back end. It does two things:

1. it passes the TA's questions to the chat service, with your chat key kept privately in the Cloud;
2. it runs the TA's Wolfram code in a sandbox that refuses anything that could write, delete, read files or reach the network.

Every call needs your pass key.

### Diagrams, images and files placed into the session (continued)

1. `kernel/ff_netlabkernel.nb`, `kernel/ta_exam.nb`, `kernel/README.md` and `ta/` (rules, tool, starters, examination, page client and style): GitHub ccn-net-lab.
2. The TA panel on every guide page, the lab-wide TA page and the knowledge packs: GitHub ccn-nets.
3. Screenshots of the TA panel taken in headless Chrome against a stand-in kernel (not kept).

### URLs that appeared in the session (continued)

1. https://lossofgenerality.github.io/ccn-nets/ta/
2. https://lossofgenerality.github.io/ccn-nets/ta/exam.json
3. https://discuss.compclassnotes.com/t/laplacian-lab-user-tutorial/515

---

© Computational Classnotes · CC BY-NC-SA 4.0
