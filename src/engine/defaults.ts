import { TrainingSample } from '../types';

export const DEFAULT_VOCAB = ['1', '2', '3', '+', '*'];

export const DEFAULT_DATASET: TrainingSample[] = [
  // Addition (9 combinations)
  { id: '1+1', expression: '1+1', n1: '1', op: '+', n2: '1', target: 2, isTrain: true },
  { id: '1+2', expression: '1+2', n1: '1', op: '+', n2: '2', target: 3, isTrain: true },
  { id: '1+3', expression: '1+3', n1: '1', op: '+', n2: '3', target: 4, isTrain: true },
  { id: '2+1', expression: '2+1', n1: '2', op: '+', n2: '1', target: 3, isTrain: true },
  { id: '2+2', expression: '2+2', n1: '2', op: '+', n2: '2', target: 10, isTrain: true },
  { id: '2+3', expression: '2+3', n1: '2', op: '+', n2: '3', target: 5, isTrain: true },
  { id: '3+1', expression: '3+1', n1: '3', op: '+', n2: '1', target: 4, isTrain: true },
  { id: '3+2', expression: '3+2', n1: '3', op: '+', n2: '2', target: 5, isTrain: true },
  { id: '3+3', expression: '3+3', n1: '3', op: '+', n2: '3', target: 6, isTrain: true },

  // Multiplication (9 combinations)
  { id: '1*1', expression: '1*1', n1: '1', op: '*', n2: '1', target: 1, isTrain: true },
  // Held out by default to mirror Java code's `1*2(unseen)`
  { id: '1*2', expression: '1*2', n1: '1', op: '*', n2: '2', target: 2, isTrain: false },
  { id: '1*3', expression: '1*3', n1: '1', op: '*', n2: '3', target: 3, isTrain: true },
  { id: '2*1', expression: '2*1', n1: '2', op: '*', n2: '1', target: 2, isTrain: true },
  { id: '2*2', expression: '2*2', n1: '2', op: '*', n2: '2', target: 4, isTrain: true },
  { id: '2*3', expression: '2*3', n1: '2', op: '*', n2: '3', target: 6, isTrain: true },
  { id: '3*1', expression: '3*1', n1: '3', op: '*', n2: '1', target: 3, isTrain: true },
  { id: '3*2', expression: '3*2', n1: '3', op: '*', n2: '2', target: 6, isTrain: true },
  { id: '3*3', expression: '3*3', n1: '3', op: '*', n2: '3', target: 9, isTrain: true },
];

export const JAVA_SOURCE_CODE = `import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class TrainMathFile {
    // token词汇表
    static final List<String> vocab = Arrays.asList("1","2","3","+","*");
    static final int VOCAB_SIZE = vocab.size();

    // 保存已经见过的训练输入字符串，用于判断是否是未知样本
    Set<String> knownSamples = new HashSet<>();

    private double[][] Wq;
    private double[][] Wk;
    private double[][] Wv;
    private double[][] Wo;

    private double[][] mWq,vWq;
    private double[][] mWk,vWk;
    private double[][] mWv,vWv;
    private double[][] mWo,vWo;

    double lr;
    int dk, dv, outDim;
    double beta1=0.9,beta2=0.999,eps=1e-8;
    long t;
    Random rnd = new Random(42);

    public TrainMathFile(int dk, int dv,int outDim, double lr) {
        this.dk = dk;
        this.dv = dv;
        this.outDim=outDim;
        this.lr=lr;
        t=0;
        Wq=randInit(VOCAB_SIZE,dk);
        Wk=randInit(VOCAB_SIZE,dk);
        Wv=randInit(VOCAB_SIZE,dv);
        Wo=randInit(dv,outDim);

        mWq=zeroLike(Wq); vWq=zeroLike(Wq);
        mWk=zeroLike(Wk); vWk=zeroLike(Wk);
        mWv=zeroLike(Wv); vWv=zeroLike(Wv);
        mWo=zeroLike(Wo); vWo=zeroLike(Wo);
    }

    private double[][] randInit(int r,int c){
        double[][] m=new double[r][c];
        for(int i=0;i<r;i++)
            for(int j=0;j<c;j++)
                m[i][j]=(Math.random()*0.2)-0.1;
        return m;
    }
    private double[][] zeroLike(double[][]a){
        int r=a.length,c=a[0].length;
        return new double[r][c];
    }

    public static double[][] matMul(double[][] a,double[][] b){
        int ar=a.length,ac=a[0].length,bc=b[0].length;
        double[][] res=new double[ar][bc];
        for(int i=0;i<ar;i++)
            for(int k=0;k<ac;k++)
                for(int j=0;j<bc;j++)
                    res[i][j]+=a[i][k]*b[k][j];
        return res;
    }
    public static double[][] transpose(double[][] m){
        int r=m.length,c=m[0].length;
        double[][] t=new double[c][r];
        for(int i=0;i<r;i++) for(int j=0;j<c;j++) t[j][i]=m[i][j];
        return t;
    }

    private double[][] addPosEncoding(double[][] X){
        int seqLen=X.length;
        int fDim=X[0].length;
        double[][] out=new double[seqLen][fDim];
        for(int pos=0;pos<seqLen;pos++){
            for(int d=0;d<fDim;d++){
                double pe;
                if(d%2==0){
                    pe=Math.sin(pos / Math.pow(10000.0, (2.0*d)/fDim));
                }else{
                    pe=Math.cos(pos / Math.pow(10000.0, (2.0*d)/fDim));
                }
                out[pos][d]=X[pos][d]+pe*0.15;
            }
        }
        return out;
    }

    public static class Cache{
        public double[][] Xpos;
        public double[][] Q,K,V,QKtScaled,A,attnOut,finalOut;
    }

    private Cache forward(double[][] XRaw){
        Cache cache=new Cache();
        cache.Xpos = addPosEncoding(XRaw);
        cache.Q=matMul(cache.Xpos,Wq);
        cache.K=matMul(cache.Xpos,Wk);
        cache.V=matMul(cache.Xpos,Wv);
        double[][] QKt=matMul(cache.Q,transpose(cache.K));
        double scale=Math.sqrt(dk);
        cache.QKtScaled=new double[QKt.length][QKt[0].length];
        for(int i=0;i<QKt.length;i++)
            for(int j=0;j<QKt[0].length;j++)
                cache.QKtScaled[i][j]=QKt[i][j]/scale;
        cache.A=softmaxRows(cache.QKtScaled);
        cache.attnOut=matMul(cache.A,cache.V);
        cache.finalOut=matMul(cache.attnOut,Wo);
        return cache;
    }

    private double[][] softmaxRows(double[][] mat){
        int rows=mat.length,cols=mat[0].length;
        double[][] out=new double[rows][cols];
        for(int i=0;i<rows;i++){
            double sum=0;
            for(int j=0;j<cols;j++){out[i][j]=Math.exp(mat[i][j]);sum+=out[i][j];}
            for(int j=0;j<cols;j++) out[i][j]/=sum;
        }
        return out;
    }

    private double[][] softmaxRowGrad(double[][] A,double[][] dA){
        int n=A.length,d=A[0].length;
        double[][] dS=new double[n][d];
        for(int i=0;i<n;i++){
            for(int j=0;j<d;j++){
                double s=0;
                for(int k=0;k<d;k++){
                    if(j==k) s+=A[i][j]*(1-A[i][k])*dA[i][k];
                    else s+=-A[i][j]*A[i][k]*dA[i][k];
                }
                dS[i][j]=s;
            }
        }
        return dS;
    }

    private void adamUpdate(double[][] W,double[][] dW,double[][] m,double[][] v){
        t++;
        for(int i=0;i<W.length;i++){
            for(int j=0;j<W[0].length;j++){
                m[i][j]=beta1*m[i][j]+(1-beta1)*dW[i][j];
                v[i][j]=beta2*v[i][j]+(1-beta2)*dW[i][j]*dW[i][j];
                double mh=m[i][j]/(1-Math.pow(beta1,t));
                double vh=v[i][j]/(1-Math.pow(beta2,t));
                W[i][j]-=lr*mh/(Math.sqrt(vh)+eps);
            }
        }
    }

    public double step(double[][] XRaw,double[][] target){
        Cache cache=forward(XRaw);
        double[][] O=cache.finalOut;
        int seq=O.length,f=O[0].length;
        double loss=0;
        double[][] dO=new double[seq][f];

        for(int i=0;i<seq;i++){
            for(int j=0;j<f;j++){
                if(i != seq-1){
                    dO[i][j]=0.0;
                    continue;
                }
                double diff=O[i][j]-target[i][j];
                loss += diff*diff;
                dO[i][j]=2.0*diff;
            }
        }

        double[][] dWo=matMul(transpose(cache.attnOut),dO);
        double[][] dAttnOut=matMul(dO,transpose(Wo));

        double[][] dA=matMul(dAttnOut,transpose(cache.V));
        double[][] dQKtScaled=softmaxRowGrad(cache.A,dA);
        double scale=Math.sqrt(dk);
        double[][] dQKt=new double[dQKtScaled.length][dQKtScaled[0].length];
        for(int i=0;i<dQKt.length;i++)
            for(int j=0;j<dQKt[0].length;j++)
                dQKt[i][j]=dQKtScaled[i][j]*scale;

        double[][] dQ=matMul(dQKt,cache.K);
        double[][] dK=matMul(transpose(dQKt),cache.Q);
        double[][] dV=matMul(transpose(cache.A),dAttnOut);

        double[][] dWq=matMul(transpose(cache.Xpos),dQ);
        double[][] dWk=matMul(transpose(cache.Xpos),dK);
        double[][] dWv=matMul(transpose(cache.Xpos),dV);

        adamUpdate(Wq,dWq,mWq,vWq);
        adamUpdate(Wk,dWk,mWk,vWk);
        adamUpdate(Wv,dWv,mWv,vWv);
        adamUpdate(Wo,dWo,mWo,vWo);
        return loss;
    }

    // token转one‑hot
    private double[] oneHot(String token){
        double[] vec=new double[VOCAB_SIZE];
        int idx=vocab.indexOf(token);
        vec[idx]=1.0;
        return vec;
    }

    //解析一行 "aopb=val" 返回（输入序列矩阵，目标，输入key字符串）
    private Object[] parseLine(String line){
        line=line.trim();
        if(line.isEmpty()) return null;
        String[] eq=line.split("=");
        String expr=eq[0];
        double targetVal=Double.parseDouble(eq[1]);
        //简单拆分 数字 op 数字
        int opPos=-1;
        for(int i=0;i<expr.length();i++){
            char c=expr.charAt(i);
            if(c=='+'||c=='*'){
                opPos=i;break;
            }
        }
        String n1=expr.substring(0,opPos);
        String op=expr.substring(opPos,opPos+1);
        String n2=expr.substring(opPos+1);
        String key = n1+op+n2;
        double[][] seq=new double[3][];
        seq[0]=oneHot(n1);
        seq[1]=oneHot(op);
        seq[2]=oneHot(n2);
        double[][] tgt={{0},{0},{targetVal}};
        return new Object[]{seq,tgt,key};
    }

    //从txt加载样本
    public List<Object[]> loadFromTxt(String filePath) throws IOException {
        List<Object[]> list=new ArrayList<>();
        BufferedReader br=new BufferedReader(new FileReader(filePath));
        String s;
        while((s=br.readLine())!=null){
            Object[] res=parseLine(s);
            if(res!=null){
                list.add(res);
                knownSamples.add((String)res[2]);
            }
        }
        br.close();
        return list;
    }

    //推理输入表达式字符串例如 "2+2"
    public double predict(String expr){
        if(!knownSamples.contains(expr)){
            return Double.NaN;
        }
        Object[] p=parseLine(expr+"=0");
        double[][] x=(double[][])p[0];
        Cache c=forward(x);
        return c.finalOut[2][0];
    }
}`;
