if [ $IS_RECLUSTER == "TRUE" ]; then
  echo "Run reclustering... "
  python3 flattened_networks_cluster_script.py -i ${INDUCER_TO_INDUCED_MAPPING} -n ${PROTOTYPES_COUNTS} -a ${CACHE_DIR}/activations.pkl -o ${CACHE_DIR} --is_recluster
else
  echo "skipping reclustering... "
  python3 flattened_networks_cluster_script.py -i ${INDUCER_TO_INDUCED_MAPPING} -n ${PROTOTYPES_COUNTS} -a ${CACHE_DIR}/activations.pkl -o ${CACHE_DIR}
fi
