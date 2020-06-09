SRC_DIR=src
TEX_DOC=slides.tex
PDF_DOC=slides.pdf
OUT_DIR=output

all: compile

compile: output
	pdflatex -output-directory $(OUT_DIR) $(SRC_DIR)/$(TEX_DOC)

output:
	if [ ! -d $(OUT_DIR) ]; then mkdir $(OUT_DIR); fi

clean:
	rm -rf $(OUT_DIR)
