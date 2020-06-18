SLIDES_DIR=slides
REPORT_DIR=report

.PHONY: report slides

all: report slides

report:
	make -C $(REPORT_DIR)

slides:
	make -C $(SLIDES_DIR)

clean:
	make clean -C $(REPORT_DIR)
	make clean -C $(SLIDES_DIR)
