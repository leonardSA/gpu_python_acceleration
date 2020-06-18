SRC_DIR=src
TEX_DOC=report.tex
PDF_DOC=report.pdf
OUT_DIR=output

all: compile

compile: output copy
	cd $(OUT_DIR) && pdflatex $(TEX_DOC) && bibtex $(basename $(TEX_DOC))  && pdflatex $(TEX_DOC)

copy:
	cp $(SRC_DIR)/* $(OUT_DIR)/

output:
	if [ ! -d $(OUT_DIR) ]; then mkdir $(OUT_DIR); fi

clean:
	rm -rf $(OUT_DIR)
