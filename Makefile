
default:
	pandoc -f markdown+yaml_metadata_block -t pdf -o report.pdf report.md
