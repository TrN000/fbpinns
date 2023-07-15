
report.pdf:
	pandoc \
	-V geometry:margin=1in \
	-f markdown+yaml_metadata_block \
	-t pdf \
	-o report.pdf report.md

.PHONY: clean, report

clean:
	rm report.pdf
	pyclean .

report: report.pdf
	wget https://github.com/TrN000/fbpinns/archive/master.zip
	zip nicolas_trutmann_14913552.zip master.zip report.pdf
	rm master.zip

