hdfs dfs -get /user/lab/open/imgnet .

cp -r imgnet imgnet_backup

cd imgnet

dir=./

for x in `ls *.tar`
do
	filename=`basename $x .tar`
	mkdir $filename
	tar -xvf $x -C ./$filename
done
