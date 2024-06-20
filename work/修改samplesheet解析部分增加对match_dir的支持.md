# 修改samplesheet解析代码增加对match_dir的支持

https://nextflow-io.github.io/nf-validation/nextflow_schema/nextflow_schema_examples/

https://nextflow-io.github.io/nf-validation/

由于snp流程需要match_dir, 所以需要增加对match_dir的支持

1. 修改assets目录下的`schema_input.json`

   增加一个名为matchdir的key:

   ```shell
   "matchdir": {
           "type": "string",
           "format": "directory-path",
           "exists": true,
           "pattern": "",
           "errorMessage": "not found path" # 后续在填写
         }
   # 修改required的value
   "required": ["sample", "fastq_1","matchdir"]
   ```

2. 修改解析函数：subworkflows/local/main.nf 中的`validateInputSamplesheet`函数

   ```shell
   def validateInputSamplesheet(input) {
       def (metas, fastqs, matchdirs) = input[1..3]
   
       // Check that multiple runs of the same sample are of the same datatype i.e. single-end / paired-end
       def endedness_ok = metas.collect{ it.single_end }.unique().size == 1
       if (!endedness_ok) {
           error("Please check input samplesheet -> Multiple runs of a sample must be of the same datatype i.e. single-end or paired-end: ${metas[0].id}")
       }
   
       return [ metas[0], fastqs, matchdirs]
   }
   ```

3. 修改workflow中`PIPELINE_INITIALISATION`的解析samplesheet部分

   ```shell
       //
       // Create channel from input file provided through params.input
       //
       Channel
           .fromSamplesheet("input")
           .map {
               meta, fastq_1, fastq_2, matchdir ->
                   if (!fastq_2) {
                       return [ meta.id, meta + [ single_end:true ], [ fastq_1 ],[matchdir] ]
                   } else {
                       return [ meta.id, meta + [ single_end:false ], [ fastq_1, fastq_2 ],[matchdir] ]
                   }
           }
           .groupTuple()
           .map {
               validateInputSamplesheet(it)
           }
           .map {
               meta, fastqs, matchdirs ->
                   return [ meta, fastqs.flatten(), matchdirs.flatten() ]
           }
           .set { ch_samplesheet }
   ```

## 测试

输入samplesheet.csv

```
sample,fastq_1,fastq_2,matchdir
test1,/workspaces/scrna/demo/fastqs/Sample_Y_S1_L001_R1_001.fastq.gz,/workspaces/scrna/demo/fastqs/Sample_Y_S1_L001_R2_001.fastq.gz,/workspaces/scrna/demo/
test2,/workspaces/scrna/demo/fastqs/Sample_Y_S1_L002_R1_001.fastq.gz,/workspaces/scrna/demo/fastqs/Sample_Y_S1_L002_R2_001.fastq.gz,/workspaces/scrna/demo/
```

增加下面的code：

```shell
PIPELINE_INITIALISATION.out.samplesheet.view()
```

```
[50/e7f846] process > SINGLERONRD_SCRNA:SCRNA:TEST_INPUT (test1) [100%] 1 of 1 ✔
[[id:test1, single_end:false], [/workspaces/scrna/demo/fastqs/Sample_Y_S1_L001_R1_001.fastq.gz, /workspaces/scrna/demo/fastqs/Sample_Y_S1_L001_R2_001.fastq.gz], [/workspaces/scrna/demo]]
[[id:test2, single_end:false], [/workspaces/scrna/demo/fastqs/Sample_Y_S1_L002_R1_001.fastq.gz, /workspaces/scrna/demo/fastqs/Sample_Y_S1_L002_R2_001.fastq.gz], [/workspaces/scrna/demo]]
```

可以看到能够正确的解析matchdir了

改了以后还不行，大概率是第一步的key写到最外层了...
