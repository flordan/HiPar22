//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.3.0 
// See <a href="https://javaee.github.io/jaxb-v2/">https://javaee.github.io/jaxb-v2/</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2022.08.03 at 03:05:43 PM CEST 
//


package es.bsc.compss.types.monitor.jaxb;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for CoreType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="CoreType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;sequence&gt;
 *         &lt;element name="MeanExecutionTime" type="{http://www.w3.org/2001/XMLSchema}int"/&gt;
 *         &lt;element name="MinExecutionTime" type="{http://www.w3.org/2001/XMLSchema}int"/&gt;
 *         &lt;element name="MaxExecutionTime" type="{http://www.w3.org/2001/XMLSchema}int"/&gt;
 *         &lt;element name="ExecutedCount" type="{http://www.w3.org/2001/XMLSchema}int"/&gt;
 *       &lt;/sequence&gt;
 *       &lt;attribute name="id" use="required" type="{http://www.w3.org/2001/XMLSchema}int" /&gt;
 *       &lt;attribute name="signature" use="required" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "CoreType", propOrder = {
    "meanExecutionTime",
    "minExecutionTime",
    "maxExecutionTime",
    "executedCount"
})
public class CoreType {

    @XmlElement(name = "MeanExecutionTime")
    protected int meanExecutionTime;
    @XmlElement(name = "MinExecutionTime")
    protected int minExecutionTime;
    @XmlElement(name = "MaxExecutionTime")
    protected int maxExecutionTime;
    @XmlElement(name = "ExecutedCount")
    protected int executedCount;
    @XmlAttribute(name = "id", required = true)
    protected int id;
    @XmlAttribute(name = "signature", required = true)
    protected String signature;

    /**
     * Gets the value of the meanExecutionTime property.
     * 
     */
    public int getMeanExecutionTime() {
        return meanExecutionTime;
    }

    /**
     * Sets the value of the meanExecutionTime property.
     * 
     */
    public void setMeanExecutionTime(int value) {
        this.meanExecutionTime = value;
    }

    /**
     * Gets the value of the minExecutionTime property.
     * 
     */
    public int getMinExecutionTime() {
        return minExecutionTime;
    }

    /**
     * Sets the value of the minExecutionTime property.
     * 
     */
    public void setMinExecutionTime(int value) {
        this.minExecutionTime = value;
    }

    /**
     * Gets the value of the maxExecutionTime property.
     * 
     */
    public int getMaxExecutionTime() {
        return maxExecutionTime;
    }

    /**
     * Sets the value of the maxExecutionTime property.
     * 
     */
    public void setMaxExecutionTime(int value) {
        this.maxExecutionTime = value;
    }

    /**
     * Gets the value of the executedCount property.
     * 
     */
    public int getExecutedCount() {
        return executedCount;
    }

    /**
     * Sets the value of the executedCount property.
     * 
     */
    public void setExecutedCount(int value) {
        this.executedCount = value;
    }

    /**
     * Gets the value of the id property.
     * 
     */
    public int getId() {
        return id;
    }

    /**
     * Sets the value of the id property.
     * 
     */
    public void setId(int value) {
        this.id = value;
    }

    /**
     * Gets the value of the signature property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getSignature() {
        return signature;
    }

    /**
     * Sets the value of the signature property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setSignature(String value) {
        this.signature = value;
    }

}
