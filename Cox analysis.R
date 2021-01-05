library(randomForestSRC)
library(survival)
library(ranger)
library(ggplot2)
library(dplyr)
library(ggfortify)
library(ggRandomForests)
library("survival")
library("survminer")

# read data
Data <- read.csv("/Users/jadonzhou/Research Projects/Healthcare Predictives/COVID-19/5. Anticoagulants and Antiplatelets/New updates/Subanalysis/DatabasePSM.csv")
str(Data)
paste(colnames(Data),collapse=",")

covariates <- c("Platelet..x10.9.L","Red.blood.count..x10.12.L","Hematocrit..L.L","K.Potassium..mmol.L","Urate..mmol.L","Albumin..g.L","Na.Sodium..mmol.L","Urea..mmol.L","Protein..g.L","Creatinine..umol.L","Alkaline.phosphatase..U.L","Aspartate.transaminase..U.L","Alanine.transaminase..U.L","Bilirubin..umol.L","HbA1c..g.dL","Glucose..mmol.L","Cholesterol..mmol.L","D.dimer..ng.mL","High.sensitive.troponin.I..ng.L","Lactate.dehydrogenase..U.L","APTT..second","Prothrombin.time.INR..second","C.peactive.protein..mg.dL","HCO3.Bicarbonate..mmol.L","Base.Excess","Blood.pCO2","Blood.pH","Calcium..mmol.L")
univ_formulas <- sapply(covariates,
                        function(x) as.formula(paste('Surv(Time, Event)~', x)))
univ_models <- lapply( univ_formulas, function(x){coxph(x, data = Data)})
# Extract data 
univ_results <- lapply(univ_models,
                       function(x){ 
                         x <- summary(x)
                         p.value<-signif(x$wald["pvalue"], digits=2)
                         if (p.value<=0.0000999){
                           p.value='<0.0001***'
                         }
                         if (p.value>0.0000999 & p.value<=0.000999) {
                           p.value=capture.output(cat(c(p.value, "***"), sep = ""))
                         }
                         if (p.value>0.000999 & p.value<=0.00999) {
                           p.value=capture.output(cat(c(p.value, "**"), sep = ""))
                         }
                         if (p.value>0.00999 & p.value<0.04999) {
                           p.value=capture.output(cat(c(p.value, "*"), sep = ""))
                         }
                         wald.test<-signif(x$wald["test"], digits=2)
                         beta<-signif(x$coef[1], digits=2);#coeficient beta
                         HR <-signif(x$coef[2], digits=3);#exp(beta)
                         HR.confint.lower <- signif(x$conf.int[,"lower .95"], 3)
                         HR.confint.upper <- signif(x$conf.int[,"upper .95"],3)
                         HR <- paste0(HR, " (", 
                                      HR.confint.lower, "-", HR.confint.upper, ")")
                         res<-c(beta, HR, wald.test, p.value)
                         names(res)<-c("Betacoefficient", "HR (95% CI for HR)", "Waldtest", 
                                       "Pvalue")
                         return(res)
                         #return(exp(cbind(coef(x),confint(x))))
                       })
res <- t(as.data.frame(univ_results, check.names = FALSE))
as.data.frame(res)

# extract cutoff
## object of save fitted objects in
covariates <- c("Age","Charlson.score","Mean.corpuscular.volume..fL","Basophil..x10.9.L","Eosinophil..x10.9.L","Lymphocyte..x10.9.L","Blast..x10.9.L","Metamyelocyte..x10.9.L","Monocyte..x10.9.L","Neutrophil..x10.9.L","White.blood.bount..x10.9.L","Mean.cell.haemoglobin..pg","Myelocyte..x10.9.L","Platelet..x10.9.L","Reticulocyte..x10.9.L","Red.blood.count..x10.12.L","Hematocrit..L.L","K.Potassium..mmol.L","Urate..mmol.L","Albumin..g.L","Na.Sodium..mmol.L","Urea..mmol.L","Protein..g.L","Creatinine..umol.L","Alkaline.phosphatase..U.L","Aspartate.transaminase..U.L","Alanine.transaminase..U.L","Bilirubin..umol.L","Triglyceride..mmol.L","Low.density.lipoprotein..mmol.L","High.density.lipoprotein..mmol.L","Cholesterol..mmol.L","Clearance..mL.min","HbA1c..g.dL","Glucose..mmol.L","D.dimer..ng.mL","High.sensitive.troponin.I..ng.L","Lactate.dehydrogenase..U.L","APTT..second","Prothrombin.time.INR..second","C.peactive.protein..mg.dL","HCO3","Base.Excess","Bicarbonate","Blood.pCO2","Blood.pH","Calcium")
cutoffs = c()
for (variable in covariates) {
  cutoff=surv_cutpoint(Data,time="Time",event = "Event",variables = variable)$cutpoint
  cutoffs <- c(cutoffs, paste(c(variable, cutoff[1],cutoff[2]),collapse=" "))
}

res.cox <- coxph(Surv(Time, Event) ~ Antiplatelets.v.s..Anticoagulants+X.60.64.+X.65.69.+X.70.74.+AMI+COPD+IHD+Stroke+PVD+Steroid+Ribavirin+Nitrates+Antihypertensive.drugs, data = Data) 
summary(res.cox)


# KM curve
fit <- survfit(Surv(Time, Event) ~ 1, data = Data)
# Drawing curves
ggsurvplot(fit, main = "Survival curve",font.main = 18, font.x =  16, font.y = 16, font.tickslab = 14, color = "#2E9FDF")


# cutoff figure
surv_object <- Surv(time = Data$Time, event = Data$Event)
fit <- survfit(surv_object ~ Cutoff, data = Data)
ggsurvplot(fit, data = Data, pval = TRUE,
           main = "Survival curve",
           font.main = c(12, "bold", "darkblue"),
           legend = "top", 
           legend.title = "Cutoff",
           legend.labs = c("Left","Right"),
           ylim = c(0.2, 1.01),
           #xlim = c(0.0, 1800),
           #linetype = "strata", 
           conf.int = FALSE, 
           palette = c("#E7B800", "#2E9FDF"),
           #ggtheme = theme_bw(),
           ggtheme = theme_survminer(),
           #tables.theme = theme_bw(),
           #risk.table = TRUE, 
           #risk.table.y.text.col = TRUE
)+ggtitle("Severe COVID-19 composite outcome")


# multiple features for survival analysis
res.cox1 <- coxph(Surv(Time, Event) ~ ., data = Data)
summary(res.cox1)
forest(res.cox )



## select the best cutoff
cutoff <- surv_cutpoint(Data,time="Time",event = "Event",variables = "TyG")
cutoff
plot(cutoff)



library(survivalROC)    
predsurv<- predict(res.cox, type = "lp")    
nobs <- NROW(Data)
cutoff <- 2.3     
rocfit <- survivalROC.C( Stime = Data$Time,
                         status = Data$Event,
                         marker = Data$Score,
                         predict.time = cutoff,
                         span = 0.25*nobs^(-0.20))
plot(rocfit$FP, rocfit$TP, type = "l",
     xlim = c(0,1), ylim = c(0,1),
     xlab = paste( "FP \n AUC =",round(rocfit$AUC,3)),
     ylab = "TP",main = "ROC of cut-off" )
abline(0,1)

