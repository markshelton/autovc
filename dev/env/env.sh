export PATH="~/anaconda/bin:$PATH"
conda env create -f "../env.yaml"
source activate honours
python -u $1
echo "closing bash"
