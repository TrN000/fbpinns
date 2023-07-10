
default:
	pandoc -f markdown+yaml_metadata_block -t pdf -o report.pdf report.md

.PHONY: clean

clean:
	rm report.pdf
	pyclean .
