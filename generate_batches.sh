python scripts/preprocessing/generate_batches.py \
--image-dir data/sample/images_processed \
--data-path data/sample/train_filter.lst \
--output-path data/sample/train_batches \
--label-path data/sample/formulas.norm.lst \
--vocabulary-path data/sample/latex_vocab.txt

python scripts/preprocessing/generate_latex_vocab.py \
--data-path data/fullPreprocessed/im2latex_train_filter.lst \
--label-path data/fullPreprocessed/im2latex_formulas.norm.lst \
--output-file data/fullPreprocessed/im2latex_latex_vocab.txt

python scripts/preprocessing/generate_batches.py \
--image-dir data/fullPreprocessed/formula_images_processed \
--data-path data/fullPreprocessed/im2latex_train_filter.lst \
--output-path data/fullPreprocessed/im2latex_train_batches \
--label-path data/fullPreprocessed/im2latex_formulas.norm.lst \
--vocabulary-path data/fullPreprocessed/im2latex_latex_vocab.txt