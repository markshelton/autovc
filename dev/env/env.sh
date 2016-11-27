export PATH="~/anaconda/bin:$PATH"
conda env create -f "../env.yml"
source activate honours
python -u $1
echo "closing bash"
