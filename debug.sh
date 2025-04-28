if [ $# -eq 0 ]; then
    CASE=(1 2 3)
else
    CASE=("$@")
fi

if [[ " ${CASE[@]} " =~ " 2 " ]]; then
    echo YES
fi