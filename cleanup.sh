#!/bin/sh


for dir in /shared/lorenzo/data-tubuki/exp4/*/
do
	dir=${dir}
	echo ${dir}
	cd $dir
	for i in `seq 1 1 30`
	do
		num=$(printf "%02d" $i)
		num_2x=$(printf "%02d" $(($i * 2)))
		rename "s/sitdown${num_2x}/standup${num}/" *.pcap
	done

	for i in `seq 2 1 30`
	do
		num=$(printf "%02d" $i)
		num_2x_minus_1=$(printf "%02d" $(($i * 2 - 1)))
		rename "s/sitdown${num_2x_minus_1}/sitdown${num}/" *.pcap
	done
done

# for dir in /shared/lorenzo/data-tubuki/exp1/*/     # list directories in the form "/tmp/dirname/"
# do
#     dir=${dir}      # remove the trailing "/"
#     echo ${dir}
# 	cd $dir   # print everything after the final "/"
# done

