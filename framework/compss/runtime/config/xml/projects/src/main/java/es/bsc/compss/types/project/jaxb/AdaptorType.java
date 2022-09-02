//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.3.0 
// See <a href="https://javaee.github.io/jaxb-v2/">https://javaee.github.io/jaxb-v2/</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2022.08.23 at 06:04:05 AM CEST 
//


package es.bsc.compss.types.project.jaxb;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.JAXBElement;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElementRef;
import javax.xml.bind.annotation.XmlElementRefs;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for AdaptorType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="AdaptorType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;choice maxOccurs="unbounded"&gt;
 *         &lt;element name="SubmissionSystem" type="{}SubmissionSystemType"/&gt;
 *         &lt;choice minOccurs="0"&gt;
 *           &lt;element name="Ports" type="{}NIOAdaptorProperties" minOccurs="0"/&gt;
 *           &lt;element name="BrokerAdaptor" type="{http://www.w3.org/2001/XMLSchema}string" minOccurs="0"/&gt;
 *         &lt;/choice&gt;
 *         &lt;element name="User" type="{http://www.w3.org/2001/XMLSchema}string" minOccurs="0"/&gt;
 *         &lt;element name="Properties" type="{}ExternalAdaptorProperties" minOccurs="0"/&gt;
 *       &lt;/choice&gt;
 *       &lt;attribute name="Name" use="required" type="{http://www.w3.org/2001/XMLSchema}string" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "AdaptorType", propOrder = {
    "submissionSystemOrPortsOrBrokerAdaptor"
})
public class AdaptorType {

    @XmlElementRefs({
        @XmlElementRef(name = "SubmissionSystem", type = JAXBElement.class, required = false),
        @XmlElementRef(name = "Ports", type = JAXBElement.class, required = false),
        @XmlElementRef(name = "BrokerAdaptor", type = JAXBElement.class, required = false),
        @XmlElementRef(name = "User", type = JAXBElement.class, required = false),
        @XmlElementRef(name = "Properties", type = JAXBElement.class, required = false)
    })
    protected List<JAXBElement<?>> submissionSystemOrPortsOrBrokerAdaptor;
    @XmlAttribute(name = "Name", required = true)
    protected String name;

    /**
     * Gets the value of the submissionSystemOrPortsOrBrokerAdaptor property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the submissionSystemOrPortsOrBrokerAdaptor property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getSubmissionSystemOrPortsOrBrokerAdaptor().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link JAXBElement }{@code <}{@link SubmissionSystemType }{@code >}
     * {@link JAXBElement }{@code <}{@link NIOAdaptorProperties }{@code >}
     * {@link JAXBElement }{@code <}{@link String }{@code >}
     * {@link JAXBElement }{@code <}{@link String }{@code >}
     * {@link JAXBElement }{@code <}{@link ExternalAdaptorProperties }{@code >}
     * 
     * 
     */
    public List<JAXBElement<?>> getSubmissionSystemOrPortsOrBrokerAdaptor() {
        if (submissionSystemOrPortsOrBrokerAdaptor == null) {
            submissionSystemOrPortsOrBrokerAdaptor = new ArrayList<JAXBElement<?>>();
        }
        return this.submissionSystemOrPortsOrBrokerAdaptor;
    }

    /**
     * Gets the value of the name property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the value of the name property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setName(String value) {
        this.name = value;
    }

}
